import Foundation
import MoldClient
import Testing

@testable import Mold

/// The UAT hooks shipped in Release: all eight `MOLD_NATIVE_*` variables were
/// read with no `#if DEBUG` anywhere, so a notarized build would seed
/// machines, open sheets, preload a picture, swap its whole preferences domain
/// and draw canned fixtures for anyone who could set an environment variable
/// on it.
///
/// `NativeUAT` is the one reader and the one place the gate lives. These tests
/// pin both halves: the gate is really there, and nothing reads around it.
@MainActor
struct NativeUATTests {
    /// **Fails today**: there is no such type -- every site read
    /// `ProcessInfo.processInfo.environment["MOLD_NATIVE_…"]` directly.
    @Test func everyHookReadsThroughTheOneGate() throws {
        let declared = Set(NativeUAT.allCases.map(\.rawValue))
        #expect(declared.count == 9)  // the eight launch seeds, and the UAT script

        var offences: [String] = []
        var mentioned: Set<String> = []
        for file in try sources() {
            let text = try String(contentsOf: file, encoding: .utf8)
            for (number, line) in text.components(separatedBy: "\n").enumerated() {
                guard let name = hookName(in: line) else { continue }
                mentioned.insert(name)
                let code = line.trimmingCharacters(in: .whitespaces)
                let isComment = code.hasPrefix("//")
                let isTheGate = file.lastPathComponent == "NativeUAT.swift"
                guard !isComment, !isTheGate else { continue }
                offences.append("\(file.lastPathComponent):\(number + 1)")
            }
        }
        #expect(offences == [], "a MOLD_NATIVE_ hook read outside NativeUAT")
        #expect(mentioned.subtracting(declared) == [], "a hook NativeUAT does not declare")
    }

    /// The gate itself. In a Debug build -- which is what `make uat` and the
    /// test scheme build -- every hook answers; in Release the read is
    /// compiled out and the answer is always nothing.
    @Test func theGateIsTheBuildConfiguration() {
        let environment = Dictionary(uniqueKeysWithValues: NativeUAT.allCases.map { ($0.rawValue, "x") })
        for hook in NativeUAT.allCases {
            #if DEBUG
            #expect(hook.value(in: environment) == "x")
            #expect(hook.isSet(in: environment))
            #else
            #expect(hook.value(in: environment) == nil)
            #expect(!hook.isSet(in: environment))
            #endif
        }
    }

    /// The package-side twin: MoldClient cannot import the app, so
    /// `SecretStore` spells the same gate out, and a Release build can never
    /// be pointed at the throwaway secrets directory.
    @Test func theSecretsDirectoryTakesTheSameGate() {
        let fresh = SecretStore.applicationSupport(environment: [NativeUAT.fresh.rawValue: "1"])
        #if DEBUG
        #expect(fresh.lastPathComponent == SecretStore.freshDirectoryName)
        #else
        #expect(fresh.lastPathComponent == SecretStore.directoryName)
        #endif
    }

    // MARK: - Reading the sources

    /// The whole variable name on a line, or nothing. The prefix written on
    /// its own -- which the gate's own doc comment does -- is prose, not a
    /// hook.
    private func hookName(in line: String) -> String? {
        let prefix = "MOLD_NATIVE_"
        guard let range = line.range(of: prefix) else { return nil }
        let name = line[range.lowerBound...].prefix { $0.isLetter || $0.isNumber || $0 == "_" }
        return name.count > prefix.count ? String(name) : nil
    }

    private func sources() throws -> [URL] {
        let macos = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
        let files = FileManager.default.enumerator(at: macos.appending(path: "Sources/Mold"),
                                                   includingPropertiesForKeys: nil)
        return (files?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
    }
}
