#!/usr/bin/env bash

set -euo pipefail

export LC_ALL=C

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
output="$repo_root/reports/code-metrics.html"
report_ref="HEAD"

usage() {
  cat <<'USAGE'
Usage: code-report [--output PATH] [--ref GIT_REF]

Generate a self-contained HTML code-metrics report.

Options:
  -o, --output PATH  Output file (default: reports/code-metrics.html)
  -r, --ref GIT_REF Git revision to measure (default: HEAD)
  -h, --help        Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -o | --output)
      [ "$#" -ge 2 ] || { echo "code-report: $1 requires a path" >&2; exit 64; }
      output="$2"
      shift 2
      ;;
    -r | --ref)
      [ "$#" -ge 2 ] || { echo "code-report: $1 requires a Git ref" >&2; exit 64; }
      report_ref="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "code-report: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

for tool in git jq tar tokei; do
  command -v "$tool" >/dev/null 2>&1 \
    || { echo "code-report: missing $tool; run this command inside nix develop" >&2; exit 69; }
done

git -C "$repo_root" rev-parse --verify "$report_ref^{commit}" >/dev/null 2>&1 \
  || { echo "code-report: Git ref does not resolve to a commit: $report_ref" >&2; exit 65; }

snapshot_dir="$(mktemp -d)"
metrics_json="$snapshot_dir/metrics.json"
output_tmp=""
cleanup() {
  rm -rf -- "$snapshot_dir"
  if [ -n "$output_tmp" ] && [ -f "$output_tmp" ]; then
    rm -f -- "$output_tmp"
  fi
}
trap cleanup EXIT

# Start from an immutable Git snapshot. Generated/design-support sources are
# intentionally omitted from the maintained-code headline.
git -C "$repo_root" archive "$report_ref" -- . \
  ':(exclude)docs/design/**' \
  ':(exclude)bun.nix' \
  ':(exclude)studio/lib/generated/**' \
  ':(exclude).understand-anything/**' \
  ':(exclude).ua/**' \
  | tar -x -C "$snapshot_dir"

tokei "$snapshot_dir" \
  --hidden \
  --exclude '*.json' \
  --exclude '*.md' \
  --exclude '*.lock' \
  --exclude '*.plist' \
  --exclude '*.xml' \
  --exclude '*.storyboard' \
  --exclude '*.pbxproj' \
  --exclude '*.xcscheme' \
  --exclude '*.xcsettings' \
  --exclude '*.pro' \
  --exclude '*.properties' \
  --exclude '*.gradle' \
  --exclude '*.toml' \
  --exclude '*.yaml' \
  --exclude '*.yml' \
  --output json > "$metrics_json"

total_code="$(jq -r '.Total.code' "$metrics_json")"
total_comments="$(jq -r '.Total.comments' "$metrics_json")"
total_blanks="$(jq -r '.Total.blanks' "$metrics_json")"
total_text=$((total_code + total_comments + total_blanks))
parsed_files="$(jq -r '[to_entries[] | select(.key != "Total") | .value.reports | length] | add' "$metrics_json")"

language_code() {
  jq -r --arg language "$1" '
    def blobcode: .code + ([.blobs[]? | blobcode] | add // 0);
    [(.[$language].reports // [])[].stats | blobcode] | add // 0
  ' "$metrics_json"
}

area_code() {
  jq -r --arg marker "/$1/" '
    def blobcode: .code + ([.blobs[]? | blobcode] | add // 0);
    [
      to_entries[]
      | select(.key != "Total")
      | .value.reports[]
      | select(.name | contains($marker))
      | .stats
      | blobcode
    ] | add // 0
  ' "$metrics_json"
}

percent() {
  awk -v value="$1" -v total="$2" 'BEGIN { printf "%.1f", value * 100 / total }'
}

commify() {
  local value="$1"
  local result=""
  while [ "${#value}" -gt 3 ]; do
    result=",${value: -3}$result"
    value="${value:0:${#value}-3}"
  done
  printf '%s%s' "$value" "$result"
}

rust="$(language_code Rust)"
typescript="$(language_code TypeScript)"
vue="$(language_code Vue)"
python="$(language_code Python)"
shell="$(language_code Shell)"
major_total=$((rust + typescript + vue + python + shell))
other=$((total_code - major_total))
systems_surface=$((rust + typescript + vue))

inference="$(area_code crates/mold-inference)"
server="$(area_code crates/mold-server)"
desktop="$(area_code desktop)"
mobile="$(area_code apps/mobile)"
desktop_mobile=$((desktop + mobile))
web="$(area_code web)"
core="$(area_code crates/mold-core)"

test_metrics="$(jq -r '
  def blobcode: .code + ([.blobs[]? | blobcode] | add // 0);
  [
    to_entries[]
    | select(.key != "Total")
    | .value.reports[]
    | select(.name | test("(^|/)(test|tests|testdata|fixtures|androidTest)(/|$)|(_test|\\.test|\\.spec)\\.(rs|ts|tsx|js|vue)$"))
  ] as $reports
  | [$reports | length, ([$reports[].stats | blobcode] | add // 0)]
  | @tsv
' "$metrics_json")"
IFS=$'\t' read -r test_files test_code <<< "$test_metrics"

large_5k=0
large_10k=0
while IFS= read -r source_file; do
  lines="$(wc -l < "$source_file")"
  if [ "$lines" -ge 5000 ]; then
    large_5k=$((large_5k + 1))
  fi
  if [ "$lines" -ge 10000 ]; then
    large_10k=$((large_10k + 1))
  fi
done < <(jq -r '[to_entries[] | select(.key != "Total") | .value.reports[].name] | unique[]' "$metrics_json")

commit="$(git -C "$repo_root" rev-parse "$report_ref^{commit}")"
commit_short="${commit:0:10}"
snapshot_date="$(git -C "$repo_root" show -s --format=%cs "$commit")"
commit_count="$(git -C "$repo_root" rev-list --count "$commit")"
commit_epoch="$(git -C "$repo_root" show -s --format=%ct "$commit")"
window_start_epoch=$((commit_epoch - 90 * 24 * 60 * 60))
recent_commits="$(
  git -C "$repo_root" rev-list --count \
    --since="@$window_start_epoch" \
    --until="@$commit_epoch" \
    "$commit"
)"
tokei_version="$(tokei --version | awk '{print $1, $2}')"

mkdir -p "$(dirname "$output")"
output="$(cd "$(dirname "$output")" && pwd -P)/$(basename "$output")"
output_tmp="$(mktemp "${output}.tmp.XXXXXX")"

cat > "$output_tmp" <<EOF
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mold codebase snapshot</title>
  <style>
    :root { color-scheme: light dark; --ink:#181512; --muted:#6e675f; --line:#ddd4ca; --soft:#f1ece5; --paper:#fbf8f3; --accent:#c44920; --rust:#df7a2e; --ts:#368dd8; --vue:#43a97b; --python:#d7ad33; --shell:#8a68c6; --other:#80786f; }
    * { box-sizing:border-box; }
    body { margin:0; background:#e9e3db; color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    main { width:min(1100px,calc(100% - 32px)); margin:32px auto; padding:clamp(24px,5vw,64px); border:1px solid var(--line); background:radial-gradient(circle at 90% 3%,#f2d8cc 0,transparent 27%),var(--paper); box-shadow:0 24px 70px #4d3a2a1f; }
    .top { display:flex; justify-content:space-between; gap:20px; margin-bottom:clamp(42px,7vw,86px); color:var(--muted); font:700 11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.13em; text-transform:uppercase; }
    .brand { color:var(--ink); } .brand::before { content:""; display:inline-block; width:10px; height:10px; margin-right:10px; background:var(--accent); transform:rotate(45deg); }
    .hero { display:grid; grid-template-columns:1.35fr .65fr; gap:clamp(28px,7vw,82px); align-items:end; }
    h1 { margin:0; font-size:clamp(54px,10vw,112px); line-height:.84; letter-spacing:-.075em; font-variant-numeric:tabular-nums; }
    .kicker { margin:0 0 12px; color:var(--muted); font-size:clamp(17px,2.3vw,24px); }
    .unit { margin:20px 0 0; color:var(--muted); font:600 12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.13em; text-transform:uppercase; }
    .summary { padding-left:22px; border-left:3px solid var(--accent); color:var(--muted); font-size:14px; line-height:1.55; }
    .summary strong { display:block; margin-bottom:10px; color:var(--ink); font-size:clamp(28px,4vw,42px); line-height:1; letter-spacing:-.04em; }
    section { padding-top:34px; margin-top:34px; border-top:1px solid var(--line); }
    h2 { margin:0 0 20px; font-size:12px; letter-spacing:.14em; text-transform:uppercase; }
    .stack { display:flex; height:20px; overflow:hidden; border:1px solid var(--line); background:var(--soft); }
    .stack span { min-width:2px; }
    .langs { display:grid; grid-template-columns:repeat(6,1fr); gap:18px; margin-top:18px; }
    .lang { min-width:0; padding-top:10px; border-top:3px solid var(--other); }
    .lang b { display:block; font-size:clamp(16px,2.2vw,24px); font-variant-numeric:tabular-nums; letter-spacing:-.03em; }
    .lang small { display:block; margin-top:5px; overflow:hidden; color:var(--muted); text-overflow:ellipsis; white-space:nowrap; }
    .split { display:grid; grid-template-columns:1.3fr .7fr; gap:clamp(34px,7vw,78px); }
    .areas { display:grid; gap:16px; }
    .area-head { display:flex; justify-content:space-between; gap:14px; margin-bottom:6px; font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; }
    .track { height:8px; background:var(--soft); } .fill { height:100%; background:var(--accent); }
    .facts { display:grid; grid-template-columns:1fr 1fr; border-top:1px solid var(--line); border-left:1px solid var(--line); }
    .fact { min-height:105px; padding:17px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); }
    .fact b { display:block; font-size:clamp(26px,4vw,38px); line-height:1; letter-spacing:-.04em; font-variant-numeric:tabular-nums; }
    .fact span { display:block; margin-top:9px; color:var(--muted); font-size:11px; line-height:1.35; }
    footer { display:grid; grid-template-columns:1fr auto; gap:24px; margin-top:40px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:11px; line-height:1.55; }
    footer strong,footer code { color:var(--ink); } footer code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; } .right { text-align:right; }
    @media (max-width:760px) { main { width:100%; margin:0; border:0; box-shadow:none; } .hero,.split { grid-template-columns:1fr; } .summary { max-width:390px; } .langs { grid-template-columns:repeat(3,1fr); } footer { grid-template-columns:1fr; } .right { text-align:left; } }
    @media (prefers-color-scheme:dark) { :root { --ink:#f3ede5; --muted:#aaa198; --line:#3d3731; --soft:#28231f; --paper:#181512; } body { background:#0f0d0b; } main { background:radial-gradient(circle at 90% 3%,#4b2118 0,transparent 27%),var(--paper); box-shadow:none; } }
  </style>
</head>
<body>
<main>
  <header class="top"><span class="brand">Mold</span><span>Codebase snapshot · $snapshot_date</span></header>
  <div class="hero">
    <div><p class="kicker">A substantial, systems-heavy codebase</p><h1>$(commify "$total_code")</h1><p class="unit">maintained lines of executable source</p></div>
    <div class="summary"><strong>$(percent "$systems_surface" "$total_code")%</strong>of the maintained codebase is Rust or TypeScript/Vue: a native inference core with a large multi-surface application layer.</div>
  </div>

  <section>
    <h2>Language composition</h2>
    <div class="stack" role="img" aria-label="Code lines by language">
      <span style="width:$(percent "$rust" "$total_code")%;background:var(--rust)"></span>
      <span style="width:$(percent "$typescript" "$total_code")%;background:var(--ts)"></span>
      <span style="width:$(percent "$vue" "$total_code")%;background:var(--vue)"></span>
      <span style="width:$(percent "$python" "$total_code")%;background:var(--python)"></span>
      <span style="width:$(percent "$shell" "$total_code")%;background:var(--shell)"></span>
      <span style="width:$(percent "$other" "$total_code")%;background:var(--other)"></span>
    </div>
    <div class="langs">
      <div class="lang" style="border-color:var(--rust)"><b>$(commify "$rust")</b><small>Rust · $(percent "$rust" "$total_code")%</small></div>
      <div class="lang" style="border-color:var(--ts)"><b>$(commify "$typescript")</b><small>TypeScript · $(percent "$typescript" "$total_code")%</small></div>
      <div class="lang" style="border-color:var(--vue)"><b>$(commify "$vue")</b><small>Vue · $(percent "$vue" "$total_code")%</small></div>
      <div class="lang" style="border-color:var(--python)"><b>$(commify "$python")</b><small>Python · $(percent "$python" "$total_code")%</small></div>
      <div class="lang" style="border-color:var(--shell)"><b>$(commify "$shell")</b><small>Shell · $(percent "$shell" "$total_code")%</small></div>
      <div class="lang"><b>$(commify "$other")</b><small>Other · $(percent "$other" "$total_code")%</small></div>
    </div>
  </section>

  <section class="split">
    <div><h2>Largest product areas</h2><div class="areas">
      <div><div class="area-head"><span>mold-inference</span><b>$(commify "$inference")</b></div><div class="track"><div class="fill" style="width:100%"></div></div></div>
      <div><div class="area-head"><span>desktop + mobile</span><b>$(commify "$desktop_mobile")</b></div><div class="track"><div class="fill" style="width:$(percent "$desktop_mobile" "$inference")%"></div></div></div>
      <div><div class="area-head"><span>mold-server</span><b>$(commify "$server")</b></div><div class="track"><div class="fill" style="width:$(percent "$server" "$inference")%"></div></div></div>
      <div><div class="area-head"><span>web</span><b>$(commify "$web")</b></div><div class="track"><div class="fill" style="width:$(percent "$web" "$inference")%"></div></div></div>
      <div><div class="area-head"><span>mold-core</span><b>$(commify "$core")</b></div><div class="track"><div class="fill" style="width:$(percent "$core" "$inference")%"></div></div></div>
    </div></div>
    <div><h2>Engineering signals</h2><div class="facts">
      <div class="fact"><b>$(commify "$parsed_files")</b><span>parsed source files</span></div>
      <div class="fact"><b>$(percent "$test_code" "$total_code")%</b><span>recognizable test code · $(commify "$test_files") files</span></div>
      <div class="fact"><b>$(commify "$large_5k")</b><span>files over 5k lines · $large_10k over 10k</span></div>
      <div class="fact"><b>$(commify "$recent_commits")</b><span>commits in 90 days · $(commify "$commit_count") total</span></div>
    </div></div>
  </section>

  <footer>
    <div><strong>$tokei_version via the Mold Nix dev shell.</strong><br>Git snapshot only. Excludes JSON, documentation/design support, generated code, lockfiles, binary assets, build output, caches, downloaded models, and Understand Anything data.</div>
    <div class="right"><code>$commit_short</code><br>$(commify "$total_text") measured text lines</div>
  </footer>
</main>
</body>
</html>
EOF

mv -f -- "$output_tmp" "$output"
output_tmp=""
printf 'Code report: %s\n' "$output"
