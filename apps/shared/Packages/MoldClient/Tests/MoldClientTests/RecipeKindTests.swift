import Foundation
import Testing

@testable import MoldClient

/// What a recipe makes comes from its own profile, never a family list: the
/// checked-in generation profiles are the fixtures, so a new family is
/// classified correctly the day it ships.
struct RecipeKindTests {
    private func profiles() throws -> [GenerationProfileSet] {
        struct Document: Decodable {
            struct Row: Decodable { let profile: GenerationProfileSet }
            let profiles: [Row]
        }
        let root = try #require(RepoFixtures.repoRoot)
        let url = root.appending(path: "docs/generated/generation-profiles-v1.json")
        return try MoldJSON.decoder.decode(Document.self, from: Data(contentsOf: url)).profiles.map(\.profile)
    }

    @Test func everyKindIsRepresentedInTheShippedProfiles() throws {
        let kinds = Set(try profiles().flatMap(\.recipes).map(\.makes))
        #expect(kinds == [.picture, .clip, .mesh])
    }

    @Test func aMeshRecipeDeliversGLBAndIsNotAClip() throws {
        let meshes = try profiles().flatMap(\.recipes).filter { $0.makes == .mesh }
        #expect(!meshes.isEmpty)
        #expect(meshes.allSatisfy { $0.capabilities.output?.formats.contains("glb") == true })
    }

    @Test func everyClipRecipeHasATimeAxis() throws {
        let clips = try profiles().flatMap(\.recipes).filter { $0.makes == .clip }
        #expect(!clips.isEmpty)
        #expect(clips.allSatisfy { $0.temporal != nil })
    }
}
