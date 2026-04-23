set -euo pipefail
cd "$(dirname "$0")/.."

UD="https://raw.githubusercontent.com/UniversalDependencies"
OUT="data/treebanks"
mkdir -p "$OUT"

declare -A REPOS=(
    ["tr"]="UD_Turkish-Kenet"
    ["kk"]="UD_Kazakh-KTB"
    ["tt"]="UD_Tatar-NMCTT"
)
declare -A PREFIXES=(
    ["tr"]="tr_kenet-ud"
    ["kk"]="kk_ktb-ud"
    ["tt"]="tt_nmctt-ud"
)

for lang in "${!REPOS[@]}"; do
    repo="${REPOS[$lang]}"
    pfx="${PREFIXES[$lang]}"
    dir="$OUT/$lang"; mkdir -p "$dir"
    echo "=== $repo"
    for split in train dev test; do
        out="$dir/${lang}-ud-${split}.conllu"
        [ -f "$out" ] && echo "  $split already exists" && continue
        url="${UD}/${repo}/master/${pfx}-${split}.conllu"
        code=$(curl -s -o "$out" -w "%{http_code}" "$url")
        if [ "$code" = "200" ] && [ -s "$out" ]; then
            echo "  $split: $(grep -c "^# sent_id" "$out" 2>/dev/null || echo "?") sentences"
        else
            rm -f "$out"
            echo "  $split: not found upstream — clone full repo instead"
        fi
    done
done

echo ""
echo "=== UD Tatar-NMCTT (only test split is published on raw GitHub)"
python3 scripts/download_ud_tatar_test.py

echo ""
echo "=== KazDet (large Kazakh treebank, ~934k tokens)"
echo "    git clone https://github.com/nlacslab/kazdet data/treebanks/kk_kazdet"

echo ""
echo "=== Parallel Turkic UD (ACL 2025 — tt/tr/az/ky aligned)"
echo "    git clone https://github.com/UniversalDependencies/UD_Tatar-NMCTT data/treebanks/parallel_tt"

echo "Done."