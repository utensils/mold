import Foundation
import Testing

@testable import MoldClient

/// Pins the Swift family lists against the Rust that owns them.
///
/// `Model.isGenerator` decides what a person sees in a model picker. If mold
/// adds an auxiliary family and this app doesn't know, a ControlNet quietly
/// shows up as something you could render with. Reading the Rust is cheaper
/// than remembering to update two lists.
private func rustFamilies(_ constant: String) throws -> Set<String> {
    let root = try #require(RepoFixtures.repoRoot, "mold checkout not found above the tests")
    let source = try String(contentsOf: root.appending(path: "crates/mold-core/src/manifest.rs"),
                            encoding: .utf8)
    let declaration = "pub const \(constant): &[&str] = &["
    let start = try #require(source.range(of: declaration), "\(constant) not found")
    let end = try #require(source.range(of: "];", range: start.upperBound..<source.endIndex))
    let body = source[start.upperBound..<end.lowerBound]

    var families: Set<String> = []
    // Quoted literals are the families themselves; a bare identifier is a
    // constant reference, resolved below.
    for match in body.matches(of: /"([^"]+)"/) {
        families.insert(String(match.1))
    }
    for match in body.matches(of: /\n\s*([A-Z][A-Z0-9_]+),/) {
        let name = String(match.1)
        let constDecl = try #require(source.range(of: "pub const \(name): &str = \""))
        let close = try #require(source.range(of: "\"", range: constDecl.upperBound..<source.endIndex))
        families.insert(String(source[constDecl.upperBound..<close.lowerBound]))
    }
    return families
}

@Test func utilityFamiliesMatchTheRustManifest() throws {
    #expect(try rustFamilies("UTILITY_FAMILIES") == Model.utilityFamilies)
}

@Test func upscalerFamiliesMatchTheRustManifest() throws {
    #expect(try rustFamilies("UPSCALER_FAMILIES") == Model.upscalerFamilies)
}

@Test func auxiliaryFamiliesMatchTheRustManifest() throws {
    // Includes HUNYUAN3D_PAINT_FAMILY, which is a constant reference rather
    // than a literal -- the parser above resolves it.
    #expect(try rustFamilies("AUXILIARY_FAMILIES") == Model.auxiliaryFamilies)
}

/// `Model.controlNetFamilies` names which installed models the Refine
/// group's adapter picker may offer -- there is no separate Rust
/// `CONTROLNET_FAMILIES` constant to pin against byte for byte, so the honest
/// check is that it never drifts outside the auxiliary set the test above
/// already pins. `"controlnet"` is a literal inside `AUXILIARY_FAMILIES`
/// (`manifest.rs`), not a second capability rule.
@Test func controlNetFamiliesStayInsideTheRustAuxiliarySet() throws {
    #expect(Model.controlNetFamilies.isSubset(of: try rustFamilies("AUXILIARY_FAMILIES")))
}
