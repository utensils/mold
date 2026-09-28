#!/usr/bin/env bash
# Print the UDID of an available simulator on an iOS 26+ runtime.
#
#   pick-simulator.sh [iPhone|iPad] [preferred name]
#   pick-simulator.sh small     # an iPhone SE, the smallest screen -- or
#                               # nothing, when none is installed
#
# A UDID, never a name: a hosted runner carries several iOS 26.x runtimes,
# each with its own "iPhone 17 Pro", and xcodebuild and simctl would each
# resolve a bare name on their own -- the dark-mode audit could then run on a
# device that was never switched to dark. The preferred model wins when it
# exists; otherwise the first of the family on the newest runtime.
set -euo pipefail

family="${1:-iPhone}"
if [ "$family" = small ]; then
  xcrun simctl list devices available -j | ruby -rjson -e '
    JSON.parse(STDIN.read)["devices"]
      .select { |rt, _| rt =~ /SimRuntime\.iOS-(\d+)/ && $1.to_i >= 26 }
      .flat_map { |_, devs| devs }
      .find { |d| d["deviceTypeIdentifier"].to_s.include?("iPhone-SE") }
      .then { |d| puts d["udid"] if d }
  '
  exit 0
fi
case "$family" in
iPhone) preferred="${2:-iPhone 17 Pro}" ;;
iPad) preferred="${2:-iPad Pro 13-inch (M5)}" ;;
*) echo "pick-simulator.sh: family must be iPhone or iPad" >&2; exit 2 ;;
esac

xcrun simctl list devices available -j | ruby -rjson -e '
  family, preferred = ARGV
  runtimes = JSON.parse(STDIN.read)["devices"]
    .select { |rt, _| rt =~ /SimRuntime\.iOS-(\d+)/ && $1.to_i >= 26 }
    .sort_by { |rt, _| rt.scan(/\d+/).map(&:to_i) }.reverse
  devices = runtimes.flat_map { |_, devs| devs }.select { |d| d["name"].start_with?(family) }
  abort("no #{family} simulator on an iOS 26+ runtime") if devices.empty?
  chosen = devices.find { |d| d["name"] == preferred } || devices.first
  puts chosen["udid"]
' "$family" "$preferred"
