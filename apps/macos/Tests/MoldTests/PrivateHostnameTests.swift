import Foundation
import Testing

/// A private machine's hostname used to be scattered through the repo as the
/// "second host" example: fixture filenames and bodies, `let x = machine(…)`
/// test doubles, prose comments in `Sources`. The owner's rule: that name
/// must never appear in code again, in any form -- not a UI placeholder, not
/// a comment, not a test-host label, not a fixture filename or body, not a
/// test name. `workstation` is the neutral stand-in everywhere that example
/// needs a second host.
///
/// This walks every source and test file this app ships -- the app target,
/// both packages' `Sources` and `Tests` -- and fails if the retired name
/// appears anywhere, in any case. Spelling it out here would itself be an
/// offence, so it is built from two halves at runtime instead.
struct PrivateHostnameTests {
    private static let retiredHostname = "pl" + "ato"

    @Test func theRetiredHostnameNeverAppearsInSourceOrTestFiles() throws {
        let files = try scannedFiles()
        #expect(files.count > 100, "the scan should cover the app target plus both packages")

        var offences: [String] = []
        for file in files {
            guard let text = try? String(contentsOf: file, encoding: .utf8) else { continue }
            for (number, line) in text.components(separatedBy: "\n").enumerated()
            where line.range(of: Self.retiredHostname, options: .caseInsensitive) != nil {
                offences.append("\(file.path(percentEncoded: false)):\(number + 1)")
            }
        }
        #expect(offences == [], "the retired private hostname must not appear in code: \(offences)")
    }

    // MARK: - Reading the sources

    private func scannedFiles() throws -> [URL] {
        let macos = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos

        var roots = [
            macos.appending(path: "Sources"),
            macos.appending(path: "Tests"),
        ]
        let packages = macos.appending(path: "Packages")
        let names = (try? FileManager.default.contentsOfDirectory(at: packages, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
        for package in names {
            roots.append(package.appending(path: "Sources"))
            roots.append(package.appending(path: "Tests"))
        }

        var files: [URL] = []
        for root in roots {
            guard let enumerator = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: [.isRegularFileKey]
            ) else { continue }
            for case let url as URL in enumerator {
                let isFile = (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile ?? false
                if isFile { files.append(url) }
            }
        }
        return files
    }
}
