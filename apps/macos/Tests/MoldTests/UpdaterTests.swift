import Foundation
import Testing

@testable import Mold

/// The updater's three decisions: which feed, whether there is an updater at
/// all, and where the menu item is declared.
///
/// **Fails today**: none of these types exist. Mold had no updater, so a
/// signed build was a dead end -- the only way to a newer one was to notice
/// and download it.
@MainActor
struct UpdaterTests {
    // MARK: - The feed allowlist

    /// The same shape `endpoints_are_fixed_https_allowlist` pins for the
    /// Tauri app (`desktop/src-tauri/src/updater.rs:717-724`): an appcast can
    /// only ever be https, under this repository's releases, and an `.xml`
    /// document. A feed is the thing that tells Sparkle what to install, so a
    /// typo that moved one off this host is the whole attack.
    @Test func theFeedsAreAFixedHTTPSAllowlist() {
        for channel in UpdateChannel.allCases {
            let feed = UpdateFeed.url(for: channel)
            #expect(feed.hasPrefix("https://github.com/utensils/mold/releases/"))
            #expect(feed.hasSuffix(".xml"))
        }
        // And there really are two of them: a mapping that answered the same
        // URL for both would satisfy every assertion above.
        #expect(UpdateFeed.stable != UpdateFeed.nightly)
        #expect(UpdateChannel.allCases.count == 2)
    }

    /// The exact URLs, so a rename on the publishing side has to change this
    /// file too. They are the ones `macos-native-distribution.yml` uploads.
    @Test func theFeedsAreTheOnesTheWorkflowPublishes() {
        #expect(
            UpdateFeed.url(for: .stable)
                == "https://github.com/utensils/mold/releases/latest/download/mold-native-appcast.xml")
        #expect(
            UpdateFeed.url(for: .nightly)
                == "https://github.com/utensils/mold/releases/download/latest/mold-native-appcast-nightly.xml"
        )
    }

    /// An unknown stored value is STABLE, alone -- never nightly, never
    /// nothing. A preferences file written by a newer build, or edited by
    /// hand, must not be able to move someone onto the busier stream.
    @Test func anUnrecognisedStoredChannelIsStableAlone() {
        for stored in [nil, "", "beta", "NIGHTLY", "nightly ", "stable\n", "0"] {
            #expect(UpdateChannel(stored: stored) == .stable, "\(stored ?? "nil")")
            #expect(UpdateFeed.url(stored: stored) == UpdateFeed.stable)
        }
        #expect(UpdateChannel(stored: "nightly") == .nightly)
        #expect(UpdateFeed.url(stored: "nightly") == UpdateFeed.nightly)
    }

    /// The delegate reads the stored channel on EVERY call. Sparkle asks
    /// afresh at each check (`feedURLStringForUpdater:`), which is the whole
    /// reason the channel is chosen there rather than written into Sparkle's
    /// own defaults -- so a cached first answer would silently keep someone
    /// on the channel they left.
    @Test func theDelegateRereadsTheChannelOnEveryCheck() throws {
        let suite = "io.utensils.mold.native.updatertests"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }

        let delegate = UpdateFeedDelegate(suiteName: suite)
        #expect(delegate.currentFeedURLString() == UpdateFeed.stable)

        defaults.set("nightly", forKey: UpdateChannel.storageKey)
        #expect(delegate.currentFeedURLString() == UpdateFeed.nightly)

        // The SECOND change, back again: this is the sequence a cached
        // answer survives.
        defaults.set("stable", forKey: UpdateChannel.storageKey)
        #expect(delegate.currentFeedURLString() == UpdateFeed.stable)

        defaults.set("banana", forKey: UpdateChannel.storageKey)
        #expect(delegate.currentFeedURLString() == UpdateFeed.stable)
    }

    // MARK: - Whether there is an updater at all

    /// All eight combinations. Only a plain Release launch may update itself:
    /// a Debug build is newer than anything published, a UAT run must touch
    /// neither the network nor the real preferences, and a test host would
    /// schedule a check in the middle of a suite.
    @Test func onlyAPlainReleaseLaunchMayReplaceItself() {
        for debug in [true, false] {
            for fresh in [true, false] {
                for tests in [true, false] {
                    let enabled = UpdaterActivation.isEnabled(
                        isDebugBuild: debug, isFreshUAT: fresh, isRunningTests: tests)
                    #expect(enabled == (!debug && !fresh && !tests), "\(debug) \(fresh) \(tests)")
                }
            }
        }
    }

    /// And the gate is really wired to this process. Deliberately NOT under
    /// `#if DEBUG`: it is the one that must run in whichever configuration
    /// the suite is compiled in, like `BuildFlagsTests`.
    @Test func thisProcessHasNoUpdater() {
        #expect(SoftwareUpdates.shared == nil)
        // Two independent reasons say so here, and each must hold on its own:
        // the test-host environment, and (usually) a Debug build.
        #expect(
            UpdaterActivation.isEnabled(
                in: ["XCTestConfigurationFilePath": "/tmp/whatever"]) == false)
    }

    // MARK: - The bundle, and the one menu command

    /// The default feed baked into `Info.plist` is the stable one and the
    /// SAME string the delegate would answer for a stable user. Two copies of
    /// a URL is exactly how a channel quietly stops matching its pointer.
    @Test func theBundlesDefaultFeedIsTheStableOne() throws {
        let feed = Bundle.main.object(forInfoDictionaryKey: "SUFeedURL") as? String
        #expect(feed == UpdateFeed.stable)
        // Present, whatever it says: the placeholder is refused at release
        // time by `scripts/assert-sparkle-key.sh`, not here, because a Debug
        // build legitimately carries it.
        let key = Bundle.main.object(forInfoDictionaryKey: "SUPublicEDKey") as? String
        #expect(key?.isEmpty == false)
    }

    /// "Check for Updates…" is declared ONCE. Two menu items for one action
    /// start two update sessions, which is the same failure the README
    /// records for a shortcut bound twice.
    @Test func checkForUpdatesIsDeclaredOnce() throws {
        let files = try sources()
        // A floor, so a scan that found nothing cannot pass.
        #expect(files.count > 100)
        var declarations: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for (number, line) in text.components(separatedBy: "\n").enumerated() {
                // Prose about the item is not the item; three doc comments
                // name it, and counting those would make this pass on two
                // real declarations.
                guard line.contains("Check for Updates…"),
                      !line.trimmingCharacters(in: .whitespaces).hasPrefix("//")
                else { continue }
                declarations.append("\(file.lastPathComponent):\(number + 1)")
            }
        }
        #expect(declarations.count == 1, "\(declarations)")
    }

    private func sources() throws -> [URL] {
        let macos = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Tests/MoldTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // apps/macos
        let files = FileManager.default.enumerator(
            at: macos.appending(path: "Sources/Mold"), includingPropertiesForKeys: nil)
        return (files?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
    }
}
