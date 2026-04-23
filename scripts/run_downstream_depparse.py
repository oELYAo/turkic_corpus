#!/usr/bin/env python3
"""
UD Tatar dependency parsing with **frozen SentencePiece** subwords (transfer study).

Only `tt_nmctt-ud-test.conllu` is distributed in the UD repo; we use *k-fold*
cross-validation on that file so conditions are comparable. The parser is a
compact BiLSTM + pairwise arc scorer + deprel classifier — fixed architecture
and training budget; **only the SentencePiece model changes** per condition.

**Seeds:** `--split-seed` fixes which sentences fall in which fold (default: same
as `--seed`). `--init-seeds` repeats all folds with different `torch` / `random`
initialization (e.g. `42,43,44`) so reported std reflects optimizer noise, not
accidental split changes.

Writes: results/downstream/ud_depparse_tatar_{script}_v{vocab}.json and .csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

_SCRIPT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _SCRIPT_FILE.parent.parent
_VENV_PREFIX = (_PROJECT_ROOT / ".venv").resolve()
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
# Use sys.prefix, not sys.executable: pyenv/shims can resolve to the same base
# binary as .venv/bin/python while site-packages are still the wrong env.
if _VENV_PYTHON.is_file() and Path(sys.prefix).resolve() != _VENV_PREFIX:
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), str(_SCRIPT_FILE)] + sys.argv[1:])

try:
    import sentencepiece as spm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from conllu import parse_incr
except ModuleNotFoundError as e:
    extra = (
        f"  {_VENV_PYTHON} {_SCRIPT_FILE} …\n"
        if _VENV_PYTHON.is_file()
        else "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n"
    )
    raise SystemExit(
        f"Missing package {e.name!r}. From turkic_corpus/ install deps into .venv:\n"
        f"{extra}"
        "  Or: source .venv/bin/activate && python scripts/run_downstream_depparse.py …"
    ) from e

SCRIPT_DIR = _SCRIPT_FILE.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from normalize_script import normalize_tatar  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = _PROJECT_ROOT
EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "experiments"
UD_PATH_DEFAULT = PROJECT_ROOT / "data" / "ud" / "tt_nmctt-ud-test.conllu"
OUT_DIR = PROJECT_ROOT / "results" / "downstream"

CONDITIONS = ("tt_only", "transfer_tr", "transfer_tr_kk", "transfer_tr_kk_tt", "byte_fallback_bpe")


@dataclass
class ParsedSentence:
    token_ids: list[int]  # CoNLL-U token id column (usually 1..n)
    forms: list[str]
    upos: list[str]
    heads: list[int]  # CoNLL-U head id (0 = root) per token
    deprels: list[str]


def load_sentences(conllu_path: Path) -> list[ParsedSentence]:
    sents: list[ParsedSentence] = []
    with conllu_path.open(encoding="utf-8") as f:
        for sent in parse_incr(f):
            ids, forms, upos, heads, deprels = [], [], [], [], []
            for t in sent:
                if isinstance(t["id"], int):
                    ids.append(int(t["id"]))
                    forms.append(t["form"])
                    upos.append(t["upos"])
                    heads.append(int(t["head"]))
                    deprels.append(t["deprel"])
            if forms:
                sents.append(ParsedSentence(ids, forms, upos, heads, deprels))
    return sents


def encode_words(
    sp: spm.SentencePieceProcessor,
    forms: Sequence[str],
    *,
    transliterate: bool,
) -> tuple[list[list[int]], list[int]]:
    """Per-word SPM ids + first-subword offset per word in the LSTM sequence."""
    piece_ids: list[int] = [1]  # special ROOT uses id 1 in embedding table; 0 = pad
    first_idx: list[int] = []
    for form in forms:
        s = normalize_tatar(form) if transliterate else form
        ids = sp.encode(s, out_type=int)
        if not ids:
            ids = [sp.unk_id()]
        first_idx.append(len(piece_ids))
        piece_ids.extend(int(x) + 2 for x in ids)  # shift by 2: 0 pad, 1 root-tok
    return [piece_ids], first_idx


def head_to_candidate_index_safe(heads: list[int], token_ids: list[int]) -> list[int]:
    """Map CoNLL-U head id to arc index: 0 = ROOT, k = word index k-1 in sentence order."""
    id2wi = {tid: wi for wi, tid in enumerate(token_ids)}
    out: list[int] = []
    for h in heads:
        if h == 0:
            out.append(0)
        else:
            wi = id2wi.get(h)
            out.append(0 if wi is None else wi + 1)
    return out


class DepParser(nn.Module):
    def __init__(self, vocab_size: int, n_deprel: int, emb_dim: int, hid_dim: int, arc_dim: int):
        super().__init__()
        self.pad = 0
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dep_arc = nn.Linear(hid_dim, arc_dim)
        self.head_arc = nn.Linear(hid_dim, arc_dim)
        self.rel_mlp = nn.Linear(hid_dim * 2, n_deprel)

    def forward(self, piece_ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(piece_ids)
        o, _ = self.lstm(e)
        return o

    def arc_scores(self, H: torch.Tensor, first_pos: torch.Tensor) -> torch.Tensor:
        """H: 1,L,D. first_pos: n word first positions in L (LSTM index, not piece table)."""
        n = first_pos.size(0)
        # candidate head positions: ROOT at 0, then first piece per word
        cand = torch.cat([torch.zeros(1, dtype=torch.long, device=H.device), first_pos])
        head_h = H[0, cand]  # n+1, D
        dep_h = H[0, first_pos]  # n, D
        dh = self.dep_arc(dep_h)
        hh = self.head_arc(head_h)
        return dh @ hh.T

    def rel_logits(self, H: torch.Tensor, first_pos: torch.Tensor, head_pos: torch.Tensor) -> torch.Tensor:
        """head_pos: L index of gold/predicted head (first-of-word or 0)."""
        dep_h = H[0, first_pos]
        head_h = H[0, head_pos]
        return self.rel_mlp(torch.cat([dep_h, head_h], dim=-1))


def train_one_fold(
    model: DepParser,
    train_batches: list[tuple[torch.Tensor, list[int], list[int], list[str], list[int]]],
    dev_batch: tuple[torch.Tensor, list[int], list[int], list[str], list[int]],
    *,
    deprel_vocab: dict[str, int],
    epochs: int,
    device: torch.device,
    lr: float,
) -> DepParser:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_dev = -1.0

    for ep in range(epochs):
        model.train()
        random.shuffle(train_batches)
        total_loss = 0.0
        for piece_ids, first_pos, gold_heads, deprels, tok_ids in train_batches:
            piece_ids = piece_ids.to(device)
            n = len(first_pos)
            if n == 0:
                continue
            first_t = torch.tensor(first_pos, dtype=torch.long, device=device)
            H = model(piece_ids.unsqueeze(0))
            arc = model.arc_scores(H, first_t)
            gold = torch.tensor(gold_heads, dtype=torch.long, device=device)
            loss_arc = F.cross_entropy(arc, gold)

            head_cand_pos = torch.tensor(
                [0] + first_pos,
                dtype=torch.long,
                device=device,
            )
            gold_head_pos = head_cand_pos[gold]
            rel_logits = model.rel_logits(H, first_t, gold_head_pos)
            gold_rel = torch.tensor(
                [deprel_vocab[d] for d in deprels],
                dtype=torch.long,
                device=device,
            )
            loss_rel = F.cross_entropy(rel_logits, gold_rel)
            loss = loss_arc + loss_rel

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item())

        las, uas = evaluate_sentence(
            model,
            dev_batch[0].to(device),
            dev_batch[1],
            dev_batch[2],
            dev_batch[3],
            dev_batch[4],
            deprel_vocab,
            device,
        )
        if las + uas > best_dev:
            best_dev = las + uas
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        log.debug("epoch %d loss=%.4f dev LAS=%.3f UAS=%.3f", ep, total_loss, las, uas)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_sentence(
    model: DepParser,
    piece_ids: torch.Tensor,
    first_pos: list[int],
    gold_heads: list[int],
    deprels: list[str],
    token_ids: list[int],
    deprel_vocab: dict[str, int],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    n = len(first_pos)
    if n == 0:
        return 0.0, 0.0
    piece_ids = piece_ids.unsqueeze(0).to(device)
    H = model(piece_ids)
    first_t = torch.tensor(first_pos, dtype=torch.long, device=device)
    arc = model.arc_scores(H, first_t)
    pred_head_idx = arc.argmax(dim=-1).tolist()  # indices in [0..n]
    head_cand_pos = [0] + first_pos
    pred_head_pos = [head_cand_pos[j] for j in pred_head_idx]
    gold = torch.tensor(gold_heads, dtype=torch.long, device=device)
    pred_t = torch.tensor(pred_head_idx, dtype=torch.long, device=device)
    uas = float((pred_t == gold).sum().item() / n)

    rel_logits = model.rel_logits(H, first_t, torch.tensor(pred_head_pos, dtype=torch.long, device=device))
    pred_rel = rel_logits.argmax(dim=-1).tolist()
    gold_rel = [deprel_vocab.get(d, 0) for d in deprels]
    las = sum(
        1 for i in range(n) if pred_head_idx[i] == gold_heads[i] and pred_rel[i] == gold_rel[i]
    ) / n
    return las, uas


def build_batches_for_fold(
    sents: list[ParsedSentence],
    indices: list[int],
    sp: spm.SentencePieceProcessor,
    *,
    transliterate: bool,
    deprel_vocab: dict[str, int],
    device: torch.device,
) -> list[tuple[torch.Tensor, list[int], list[int], list[str], list[int]]]:
    batches = []
    for idx in indices:
        ps = sents[idx]
        n = len(ps.forms)
        gold_heads = head_to_candidate_index_safe(ps.heads, ps.token_ids)

        piece_rows, first_pos = encode_words(sp, ps.forms, transliterate=transliterate)
        pid = piece_rows[0]
        vocab_size = sp.get_piece_size() + 2
        t = torch.tensor(pid, dtype=torch.long, device=device)
        batches.append((t, first_pos, gold_heads, ps.deprels, ps.token_ids))

    return batches


@dataclass
class FoldResult:
    init_seed: int
    fold: int
    condition: str
    train_script: str
    vocab_size: int
    las_mean: float
    uas_mean: float
    n_test_tokens: int


def run_condition_fold(
    condition: str,
    train_script: str,
    vocab_size: int,
    fold_idx: int,
    train_idx: list[int],
    test_idx: list[int],
    sents: list[ParsedSentence],
    *,
    transliterate: bool,
    deprel_vocab: dict[str, int],
    epochs: int,
    device: torch.device,
    emb_dim: int,
    hid_dim: int,
    arc_dim: int,
    seed: int,
) -> FoldResult:
    sp_path = EXPERIMENT_ROOT / condition / train_script / f"vocab_{vocab_size}" / "spm.model"
    if not sp_path.is_file():
        raise FileNotFoundError(sp_path)
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))
    vs = sp.get_piece_size() + 2

    random.seed(seed + fold_idx)
    torch.manual_seed(seed + fold_idx)

    train_batches = build_batches_for_fold(
        sents, train_idx, sp, transliterate=transliterate, deprel_vocab=deprel_vocab, device=torch.device("cpu")
    )
    dev_batches = train_batches[-max(1, len(train_batches) // 10) :]  # 10% pseudo-dev from train
    train_batches = train_batches[: len(train_batches) - len(dev_batches)]
    if not train_batches:
        train_batches = dev_batches

    test_batches = build_batches_for_fold(
        sents, test_idx, sp, transliterate=transliterate, deprel_vocab=deprel_vocab, device=torch.device("cpu")
    )

    model = DepParser(vs, len(deprel_vocab), emb_dim, hid_dim, arc_dim).to(device)
    train_one_fold(model, train_batches, dev_batches[0], deprel_vocab=deprel_vocab, epochs=epochs, device=device, lr=1e-3)

    las_tok, uas_tok, ntok = 0.0, 0.0, 0
    for tb in test_batches:
        las, uas = evaluate_sentence(
            model, tb[0], tb[1], tb[2], tb[3], tb[4], deprel_vocab, device
        )
        n = len(tb[1])
        las_tok += las * n
        uas_tok += uas * n
        ntok += n

    return FoldResult(
        init_seed=seed,
        fold=fold_idx,
        condition=condition,
        train_script=train_script,
        vocab_size=vocab_size,
        las_mean=las_tok / ntok if ntok else 0.0,
        uas_mean=uas_tok / ntok if ntok else 0.0,
        n_test_tokens=ntok,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--ud", type=Path, default=UD_PATH_DEFAULT)
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Default split-shuffle seed; also the sole init seed unless --init-seeds is set.",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        dest="split_seed",
        help="Shuffle sentences into k folds with this RNG seed (fixed across --init-seeds). "
        "Default: same as --seed.",
    )
    p.add_argument(
        "--init-seeds",
        type=str,
        default=None,
        help="Comma-separated torch/random init seeds (e.g. 42,43,44). Each seed runs all k folds. "
        "Default: single run using --seed as init.",
    )
    p.add_argument("--emb-dim", type=int, default=120)
    p.add_argument("--hid-dim", type=int, default=200)
    p.add_argument("--arc-dim", type=int, default=100)
    p.add_argument(
        "--script",
        choices=("native", "normalized", "both"),
        default="native",
        help="Tokenizer training script + CoNLL-U FORM script; 'both' runs native then normalized.",
    )
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--results-tag",
        default="",
        help="If non-empty, output basenames get this suffix (e.g. smoke -> ud_depparse_tatar_native_v16000_smoke.json).",
    )
    args = p.parse_args()

    if not args.ud.is_file():
        raise SystemExit(f"Missing UD file {args.ud}. Run: python scripts/download_ud_tatar_test.py")

    scripts = ("native", "normalized") if args.script == "both" else (args.script,)
    for scr in scripts:
        _run_depparse_for_script_namespace(args, scr)


def _run_depparse_for_script_namespace(args: argparse.Namespace, script: str) -> None:
    device = torch.device(args.device)
    transliterate = script == "normalized"

    sents = load_sentences(args.ud)
    log.info("loaded %d sentences from %s", len(sents), args.ud)

    rels = sorted({d for s in sents for d in s.deprels})
    deprel_vocab = {r: i for i, r in enumerate(rels)}

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    if args.init_seeds:
        init_seeds = [int(x.strip()) for x in args.init_seeds.split(",") if x.strip()]
        if not init_seeds:
            raise SystemExit("--init-seeds must list at least one integer")
    else:
        init_seeds = [args.seed]

    n = len(sents)
    idxs = list(range(n))
    random.Random(split_seed).shuffle(idxs)
    fold_size = n // args.k_folds
    folds: list[tuple[list[int], list[int]]] = []
    for k in range(args.k_folds):
        start = k * fold_size
        end = n if k == args.k_folds - 1 else (k + 1) * fold_size
        test_i = idxs[start:end]
        train_i = [i for i in idxs if i not in set(test_i)]
        folds.append((train_i, test_i))

    rows: list[FoldResult] = []
    summary: dict[str, dict[str, float]] = {}

    for cond in conds:
        las_s, uas_s = [], []
        for init_s in init_seeds:
            for fi, (tr, te) in enumerate(folds):
                log.info("condition=%s script=%s init_seed=%d fold=%d", cond, script, init_s, fi)
                fr = run_condition_fold(
                    cond,
                    script,
                    args.vocab_size,
                    fi,
                    tr,
                    te,
                    sents,
                    transliterate=transliterate,
                    deprel_vocab=deprel_vocab,
                    epochs=args.epochs,
                    device=device,
                    emb_dim=args.emb_dim,
                    hid_dim=args.hid_dim,
                    arc_dim=args.arc_dim,
                    seed=init_s,
                )
                rows.append(fr)
                las_s.append(fr.las_mean)
                uas_s.append(fr.uas_mean)
        n_runs = len(las_s)
        summary[cond] = {
            "las_mean": float(sum(las_s) / n_runs),
            "las_std": float(torch.tensor(las_s).std(unbiased=False).item()) if n_runs > 1 else 0.0,
            "uas_mean": float(sum(uas_s) / n_runs),
            "uas_std": float(torch.tensor(uas_s).std(unbiased=False).item()) if n_runs > 1 else 0.0,
            "k_folds": float(args.k_folds),
            "n_init_seeds": float(len(init_seeds)),
            "n_runs": float(n_runs),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.results_tag}" if args.results_tag.strip() else ""
    base = f"ud_depparse_tatar_{script}_v{args.vocab_size}{tag}"
    meta = {
        "ud_file": str(args.ud.resolve()),
        "n_sentences": len(sents),
        "k_folds": args.k_folds,
        "split_seed": split_seed,
        "init_seeds": init_seeds,
        "epochs": args.epochs,
        "vocab_size": args.vocab_size,
        "script": script,
        "deprel_dim": len(deprel_vocab),
        "note": (
            "UD_Tatar-NMCTT currently publishes only the test split; metrics use k-fold "
            "on that file. Architecture and training steps are shared; only SentencePiece changes."
        ),
        "per_fold": [asdict(r) for r in rows],
        "summary_by_condition": summary,
    }
    json_path = OUT_DIR / f"{base}.json"
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = OUT_DIR / f"{base}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

    sum_csv = OUT_DIR / f"{base}_summary.csv"
    sum_fields = (
        "condition",
        "las_mean",
        "las_std",
        "uas_mean",
        "uas_std",
        "k_folds",
        "n_init_seeds",
        "n_runs",
    )
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_fields))
        w.writeheader()
        for c, v in summary.items():
            w.writerow({"condition": c, **v})

    log.info("Wrote %s, %s, %s", json_path, csv_path, sum_csv)


if __name__ == "__main__":
    main()
