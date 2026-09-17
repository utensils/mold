import Foundation
import Testing

@testable import MoldClient

/// CFG++ is the ONE control in this app decided by a family name, because it
/// is the one control with no capability anywhere: `generationCapabilities.ts`
/// keeps a client-side set and the server advertises nothing. A second copy of
/// a set nobody can check is exactly how two clients drift, so this reads the
/// TypeScript and fails when they do.
///
/// **Fails today** if `CFG_PLUS_FAMILIES` gains or loses a family and
/// `AdvancedControlsOffered.cfgPlusFamilies` does not: the assertion is set
/// EQUALITY, not containment, and the `count > 0` floor makes a parse that
/// found nothing a failure rather than a pass.
struct CfgPlusContractTests {
    @Test func theClientFamilySetIsStudiosOwn() throws {
        let root = try #require(RepoFixtures.repoRoot, "the mold checkout")
        let source = try String(
            contentsOf: root.appending(path: "studio/lib/generationCapabilities.ts"),
            encoding: .utf8)
        let families = try #require(Self.setLiteral(named: "CFG_PLUS_FAMILIES", in: source),
                                    "CFG_PLUS_FAMILIES in generationCapabilities.ts")
        #expect(families.count > 0, "parsed an empty set -- the literal's shape has changed")
        #expect(families == AdvancedControlsOffered.cfgPlusFamilies)
    }

    /// `const NAME = new Set([...]);` -- the one shape every family set in
    /// that file is written in. A quoted-string scan of the bracketed body, so
    /// the members are read rather than the whole file being pattern-matched.
    private static func setLiteral(named name: String, in source: String) -> Set<String>? {
        guard let declaration = source.range(of: "const \(name) = new Set([") else { return nil }
        guard let close = source.range(of: "]", range: declaration.upperBound ..< source.endIndex)
        else { return nil }
        let body = source[declaration.upperBound ..< close.lowerBound]
        var members: Set<String> = []
        var rest = body[...]
        while let open = rest.firstIndex(of: "\"") {
            let after = rest.index(after: open)
            guard let end = rest[after...].firstIndex(of: "\"") else { break }
            members.insert(String(rest[after ..< end]))
            rest = rest[rest.index(after: end)...]
        }
        return members
    }
}
