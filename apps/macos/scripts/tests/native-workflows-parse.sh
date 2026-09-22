#!/usr/bin/env bash
# Every workflow this app owns or touches must PARSE, and its inline bash must
# parse too.
#
# **Fails today**: nothing checked either. `release.yml` and `desktop.yml` run
# on `main` and on `v*` tags, and this app lives on a branch that is never
# merged -- so its jobs in them cannot be exercised here at all, and a YAML or
# shell typo would surface on somebody else's release.
#
# `bash -n` over each `run:` block is the closest thing to running them that
# does not need GitHub: it catches the unbalanced quote, the missing `fi`, the
# stray `done`.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../../../.." && pwd)"
WORKFLOWS="$HERE/.github/workflows"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FILES=(
  "$WORKFLOWS/macos-native.yml"
  "$WORKFLOWS/macos-native-distribution.yml"
  "$WORKFLOWS/release.yml"
  "$WORKFLOWS/desktop.yml"
)

for file in "${FILES[@]}"; do
  [ -f "$file" ] || { echo "FAIL: missing $file" >&2; exit 1; }
done

# 1. YAML. Ruby ships with macOS and every GitHub runner; no network, no
#    Python module to install.
ruby -ryaml -rdate -e '
  ARGV.each do |path|
    doc = YAML.safe_load(File.read(path), aliases: true, permitted_classes: [Date, Time])
    raise "#{path}: no jobs" unless doc["jobs"].is_a?(Hash) && !doc["jobs"].empty?
  end
' "${FILES[@]}" || { echo "FAIL: a workflow does not parse as YAML" >&2; exit 1; }

# 2. The jobs this app added must exist, by name, in the workflows that own
#    them -- a rename on either side is what would silently stop publishing.
ruby -ryaml -e '
  native = YAML.safe_load(File.read(ARGV[0]), aliases: true)
  release = YAML.safe_load(File.read(ARGV[1]), aliases: true)
  missing = []
  trigger = native["on"] || native[true] # YAML 1.1 treats on as a boolean.
  push = trigger.fetch("push")
  if push.key?("paths") || push.key?("paths-ignore")
    abort("native main pushes must always schedule a replacement for a stale nightly")
  end
  %w[Cargo.toml scripts/create-desktop-nightly-version.sh].each do |path|
    abort("native PR checks omit #{path}") unless trigger.fetch("pull_request").fetch("paths").include?(path)
  end
  %w[check macos-native-nightly publish-macos-native-nightly].each do |job|
    missing << "macos-native.yml:#{job}" unless native["jobs"].key?(job)
  end
  missing << "release.yml:build-macos-native-dmg" unless release["jobs"].key?("build-macos-native-dmg")
  unless release["jobs"]["release-version"]["needs"].include?("build-macos-native-dmg")
    missing << "release.yml:release-version needs build-macos-native-dmg"
  end
  # The desktop jobs must be untouched by all this.
  %w[desktop-nightly publish-desktop-nightly].each do |job|
    missing << "desktop.yml:#{job}" unless YAML.safe_load(File.read(ARGV[2]), aliases: true)["jobs"].key?(job)
  end
  abort("missing: #{missing.join(", ")}") unless missing.empty?
' "$WORKFLOWS/macos-native.yml" "$WORKFLOWS/release.yml" "$WORKFLOWS/desktop.yml" \
  || { echo "FAIL: a publishing job is missing or renamed" >&2; exit 1; }

# 3. Every `run:` block in the two workflows this app OWNS must be valid bash.
count=0
for file in "$WORKFLOWS/macos-native.yml" "$WORKFLOWS/macos-native-distribution.yml"; do
  ruby -ryaml -e '
    doc = YAML.safe_load(File.read(ARGV[0]), aliases: true)
    i = 0
    doc["jobs"].each_value do |job|
      (job["steps"] || []).each do |step|
        next unless step["run"]
        i += 1
        File.write(File.join(ARGV[1], "%s-%02d.sh" % [File.basename(ARGV[0], ".yml"), i]), step["run"])
      end
    end
  ' "$file" "$WORK"
done
for script in "$WORK"/*.sh; do
  [ -e "$script" ] || continue
  count=$((count + 1))
  bash -n "$script" || { echo "FAIL: inline bash does not parse: $script" >&2; cat "$script" >&2; exit 1; }
done
# A floor: if the extraction silently found nothing, this whole check passes
# by doing nothing.
if [ "$count" -lt 8 ]; then
  echo "FAIL: only $count run blocks extracted -- the extraction is broken" >&2
  exit 1
fi

echo "native workflows parse ok ($count run blocks)"
