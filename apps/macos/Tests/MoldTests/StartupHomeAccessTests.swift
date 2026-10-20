import Foundation
import Testing
@testable import Mold

struct StartupHomeAccessTests {
    @Test func onlyTheExistingHomeCanReceiveAccess() throws {
        let base = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let home = base.appending(path: "home")
        let other = base.appending(path: "other")
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        #expect(StartupHomeAccessPolicy.accepts(expected: home, selected: home))
        #expect(StartupHomeAccessPolicy.accepts(
            expected: URL(fileURLWithPath: home.path, isDirectory: false),
            selected: URL(fileURLWithPath: home.path, isDirectory: true)))
        #expect(!StartupHomeAccessPolicy.accepts(expected: home, selected: other))
        #expect(!StartupHomeAccessPolicy.accepts(expected: home, selected: base.appending(path: "missing")))
    }

    @Test func canonicalAliasesNameTheSameExistingHome() throws {
        let base = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let home = base.appending(path: "home")
        let alias = base.appending(path: "alias")
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: home)
        defer { try? FileManager.default.removeItem(at: base) }
        #expect(StartupHomeAccessPolicy.accepts(expected: home, selected: alias))
    }
}
