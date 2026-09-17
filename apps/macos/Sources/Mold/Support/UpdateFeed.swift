import Foundation

/// Which appcast Mold reads.
///
/// TWO FEEDS, not Sparkle's own channel tagging. Sparkle can carry several
/// channels inside ONE feed (`sparkle:channel`, `allowedChannels(for:)`), and
/// that is the right shape when a beta is a subset of the same release
/// stream. Mold's nightly is not: it is a separate, far busier stream that a
/// stable user must never be offered by accident, and the Tauri app already
/// publishes it as its own pointer under its own release
/// (`desktop/src-tauri/src/updater.rs:23-26`). Two feeds keep the two streams
/// separable at the SERVER, so a mistake in one cannot reach the other.
///
/// `nonisolated`: the project defaults to `@MainActor`, and this is read by
/// `UpdateFeedDelegate`, which Sparkle does not promise the main thread for.
nonisolated enum UpdateChannel: String, CaseIterable, Identifiable, Sendable {
    case stable
    case nightly

    /// Where the choice is remembered. `AppStorageSuite.defaults`, like every
    /// other preference -- so a `MOLD_NATIVE_FRESH` run gets a fresh one and
    /// Settings ▸ General ▸ Reset puts it back.
    static let storageKey = "updateChannel"

    /// An unknown stored value is STABLE, and that is the whole rule: a
    /// preferences file written by a newer build, a hand-edited plist, or a
    /// channel mold no longer publishes must all land on the conservative
    /// stream rather than on nothing at all.
    init(stored: String?) {
        self = UpdateChannel(rawValue: stored ?? "") ?? .stable
    }

    var id: String { rawValue }

    var label: String {
        switch self {
        case .stable: "Stable"
        case .nightly: "Nightly"
        }
    }
}

/// The two appcast URLs, and the only two Mold will ever read.
///
/// GitHub Releases only -- no gh-pages, no utensils.io -- mirroring the Tauri
/// app's two manifests exactly. `UpdateFeedTests` pins both against the same
/// allowlist `endpoints_are_fixed_https_allowlist` pins the desktop pair with
/// (`desktop/src-tauri/src/updater.rs:717-724`): https, this repository's
/// releases, an `.xml` document. `scripts/assert-sparkle-key.sh` pins the
/// stable one again, in the built bundle, where a build setting could still
/// have changed it.
nonisolated enum UpdateFeed {
    /// `releases/latest/download/…` follows whatever the newest non-prerelease
    /// tag is, so a stable user needs no pointer flip at all.
    static let stable =
        "https://github.com/utensils/mold/releases/latest/download/mold-native-appcast.xml"
    /// `releases/download/latest/…` is the rolling `latest` PRE-release, which
    /// is where nightly assets live.
    static let nightly =
        "https://github.com/utensils/mold/releases/download/latest/mold-native-appcast-nightly.xml"

    static func url(for channel: UpdateChannel) -> String {
        switch channel {
        case .stable: stable
        case .nightly: nightly
        }
    }

    /// What the delegate asks: a stored preference straight to a feed, with
    /// the unknown-value rule applied once, here.
    static func url(stored: String?) -> String {
        url(for: UpdateChannel(stored: stored))
    }
}
