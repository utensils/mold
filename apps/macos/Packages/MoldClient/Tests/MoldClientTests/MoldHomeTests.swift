import Foundation
import Testing

@testable import MoldClient

// The Swift twin of `Config::mold_dir` in `crates/mold-core/src/config.rs`.
// The embedded engine used to read `MOLD_HOME` or fall straight back to
// `~/.mold`, so an app launched from Finder -- which hands it no environment at
// all -- ran against a different home than the CLI, the Tauri app and every
// `mold` command on the same Mac: no models, an empty library, and settings
// nobody recognised.

private let home = URL(filePath: "/Users/tester")

@Test func anExplicitMoldHomeWinsOverEverything() throws {
    let pointer = URL.temporaryDirectory.appending(path: "pointer-\(UUID().uuidString)")
    try "/Volumes/External/mold2".write(to: pointer, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: pointer) }

    let resolved = MoldHome.resolve(environment: [
        "MOLD_HOME": "/tmp/uat-home",
        "MOLD_HOME_POINTER_PATH": pointer.path,
    ], home: home)
    #expect(resolved.url.path == "/tmp/uat-home")
    #expect(resolved.source == .environment)
}

@Test func followsTheBootstrapPointerTheRestOfMoldWrites() throws {
    let pointer = URL.temporaryDirectory.appending(path: "pointer-\(UUID().uuidString)")
    try "/Volumes/External/mold2\n".write(to: pointer, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: pointer) }

    let resolved = MoldHome.resolve(environment: ["MOLD_HOME_POINTER_PATH": pointer.path],
                                    home: home)
    #expect(resolved.url.path == "/Volumes/External/mold2")
    #expect(resolved.source == .saved)
}

/// On macOS the pointer lives beside the rest of mold's own state, because
/// `dirs::config_dir()` is `~/Library/Application Support` here -- NOT
/// `~/.config`, which is what the same call returns on Linux.
@Test func looksForThePointerWhereTheRustSideWritesIt() {
    #expect(MoldHome.pointerPath(environment: [:], home: home).path
            == "/Users/tester/Library/Application Support/mold/home")
}

@Test(arguments: ["", "   ", "relative/path"])
func ignoresAPointerThatIsNotAnAbsolutePath(contents: String) throws {
    let pointer = URL.temporaryDirectory.appending(path: "pointer-\(UUID().uuidString)")
    try contents.write(to: pointer, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: pointer) }

    let resolved = MoldHome.resolve(environment: ["MOLD_HOME_POINTER_PATH": pointer.path],
                                    home: home)
    #expect(resolved.url.path == "/Users/tester/.mold")
    #expect(resolved.source == .fallback)
}

@Test func fallsBackToTheDefaultRootWithNoPointerAtAll() {
    let resolved = MoldHome.resolve(
        environment: ["MOLD_HOME_POINTER_PATH": "/nowhere/\(UUID().uuidString)"],
        home: home
    )
    #expect(resolved.url.path == "/Users/tester/.mold")
    #expect(resolved.source == .fallback)
}

/// A saved home that is not there means an external drive is unplugged, not a
/// new home to create -- mold's own `ensure_saved_mold_dir_available` refuses
/// for the same reason. Starting an engine on a freshly made empty directory
/// looks exactly like losing every model and every print.
@Test func aChosenHomeThatIsNotThereIsReportedRatherThanRecreated() throws {
    let pointer = URL.temporaryDirectory.appending(path: "pointer-\(UUID().uuidString)")
    let missing = "/Volumes/Unplugged/mold2"
    try missing.write(to: pointer, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: pointer) }

    let resolved = MoldHome.resolve(environment: ["MOLD_HOME_POINTER_PATH": pointer.path],
                                    home: home)
    #expect(resolved.unavailableReason?.contains(missing) == true)

    // An explicit MOLD_HOME names a directory the caller means to create, so it
    // is never refused -- that is what the UAT command relies on.
    let uat = MoldHome.resolve(environment: ["MOLD_HOME": "/tmp/uat-\(UUID().uuidString)"],
                               home: home)
    #expect(uat.unavailableReason == nil)
}
