import Foundation
import Testing

@testable import Mold

/// 05-M13: the button said it resets "every pane's own sort and scope … and
/// the remembered machine", and four persisted keys were in neither
/// `PreferencesReset.keys` nor its written-down exclusions -- so after it, the
/// Library was still scoped to a collection and Generate still targeted the
/// last machine. The old test only asserted the LISTED keys were removed,
/// which is exactly the shape of assertion that cannot see an omission.
///
/// So this one reads the app's own sources instead: every key written to
/// `AppStorageSuite.defaults` must be named by one list or the other.
@MainActor
struct PreferencesResetTests {
    @Test func preferencesKeepOneObjectIdentityForTheirObservers() {
        #expect(AppStorageSuite.defaults === AppStorageSuite.defaults)
    }

    @Test func everyPersistedPreferenceIsEitherResetOrDeliberatelyKept() throws {
        let written = try Self.persistedKeys()
        #expect(written.count > 20, "the scan found almost nothing -- it has stopped working")

        let named = Set(PreferencesReset.keys).union(PreferencesReset.kept)
        #expect(written.subtracting(named).sorted() == [],
                "a key nothing has decided about: add it to keys or to kept")
        #expect(named.subtracting(written).sorted() == [],
                "a key nothing writes any more")
        #expect(Set(PreferencesReset.keys).intersection(PreferencesReset.kept) == [])
        #expect(PreferencesReset.keys.count == Set(PreferencesReset.keys).count)
    }

    /// **Fails today**: `libraryScope`, `libraryEdge`, `generateMachine` and
    /// `defaultMachine` survived a reset.
    @Test func theFourTheButtonUsedToMissAreCleared() {
        let name = "io.utensils.mold.native.tests.reset.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        let missed = ["libraryScope", "libraryEdge", "generateMachine", "defaultMachine"]
        for key in missed + Array(PreferencesReset.kept) { defaults.set("stale", forKey: key) }

        PreferencesReset.reset(in: defaults)

        for key in missed { #expect(defaults.object(forKey: key) == nil, "\(key) survived") }
        for key in PreferencesReset.kept {
            #expect(defaults.string(forKey: key) == "stale", "\(key) should have been left alone")
        }
    }

    // MARK: - Reading the sources

    /// Every string this app uses as a `UserDefaults` key against the app's
    /// own suite. The same "parse the real source rather than compare two
    /// client constants" idiom `SettingKeysContractTests` uses against
    /// `config_keys.rs`.
    ///
    /// An argument it cannot resolve to a literal is a FAILURE, not a skip:
    /// a key spelled some third way is precisely what this test exists to
    /// catch, so a new one has to be a literal or a `static let`.
    static func persistedKeys() throws -> Set<String> {
        var keys: Set<String> = []
        for file in try sources() {
            let text = try String(contentsOf: file, encoding: .utf8)
            // `PreferencesReset` is the resetter; its `forKey: key` is this
            // list's own loop variable.
            guard file.lastPathComponent != "PreferencesReset.swift" else { continue }
            let constants = constantStrings(in: text)
            for reference in references(in: text) {
                if let literal = reference.literal {
                    keys.insert(literal)
                } else if let name = reference.name,
                          let resolved = constants[name] ?? globalConstants[name] {
                    keys.insert(resolved)
                } else {
                    Issue.record("\(file.lastPathComponent): unresolved defaults key \(reference)")
                }
            }
        }
        return keys
    }

    private static let globalConstants: [String: String] = {
        var all: [String: String] = [:]
        for file in (try? sources()) ?? [] {
            guard let text = try? String(contentsOf: file, encoding: .utf8) else { continue }
            all.merge(constantStrings(in: text)) { first, _ in first }
        }
        return all
    }()

    private static func sources() throws -> [URL] {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
            .appending(path: "Sources/Mold")
        let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)
        return (files?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
    }

    /// `static let <name> = "<literal>"`, by name.
    private static func constantStrings(in text: String) -> [String: String] {
        matches(#"static let (\w+) *= *"([^"]+)""#, in: text)
            .reduce(into: [:]) { $0[$1[0]] = $1[1] }
    }

    struct Reference: CustomStringConvertible {
        var literal: String?
        var name: String?
        var description: String { literal ?? name ?? "?" }
    }

    /// The two ways this app names a key: `@AppStorage(…, store:
    /// AppStorageSuite.defaults)`, and a `forKey:` argument on a receiver
    /// actually called `defaults`. The receiver check is what keeps a
    /// dictionary's own `removeValue(forKey:)` out of the answer.
    private static func references(in text: String) -> [Reference] {
        let storage = matches(#"@AppStorage\(([^,]+), *store: *AppStorageSuite\.defaults"#, in: text)
        let direct = matches(#"(?:AppStorageSuite\.)?\bdefaults\.\w+\([^()]*forKey: *([^,)]+)"#, in: text)
        return (storage + direct).map { groups in
            let argument = groups[0].trimmingCharacters(in: .whitespaces)
            guard argument.hasPrefix("\"") else {
                return Reference(name: argument.replacingOccurrences(of: "Self.", with: "")
                    .components(separatedBy: ".").last)
            }
            return Reference(literal: String(argument.dropFirst().dropLast()))
        }
    }

    private static func matches(_ pattern: String, in text: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let whole = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: whole).map { match in
            (1..<match.numberOfRanges).compactMap { index in
                Range(match.range(at: index), in: text).map { String(text[$0]) }
            }
        }
    }
}
