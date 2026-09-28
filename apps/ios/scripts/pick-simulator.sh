#!/usr/bin/env bash
# Print the name of an available iPhone simulator on an iOS 26+ runtime.
# Prefers `$1` (default "iPhone 17 Pro") when it exists, else the first iPhone
# on the newest iOS runtime -- a laptop and a hosted runner never carry the
# same set, and a pinned name that is missing fails the whole run.
set -euo pipefail

preferred="${1:-iPhone 17 Pro}"
xcrun simctl list devices available -j | ruby -rjson -e '
  preferred = ARGV[0]
  runtimes = JSON.parse(STDIN.read)["devices"]
    .select { |rt, _| rt =~ /SimRuntime\.iOS-(\d+)/ && $1.to_i >= 26 }
    .sort_by { |rt, _| rt.scan(/\d+/).map(&:to_i) }.reverse
  phones = runtimes.flat_map { |_, devs| devs.map { |d| d["name"] } }.grep(/^iPhone/)
  abort("no iPhone simulator on an iOS 26+ runtime") if phones.empty?
  puts(phones.include?(preferred) ? preferred : phones.first)
' "$preferred"
