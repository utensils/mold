import Foundation
import Testing
@testable import MoldClient

@Test func organizationPreservesDestinationEditsAndSuppressesOrdinaryReaddition() {
    let source = LibraryOrganizationSnapshot(collections: ["work"], tags: ["draft"], title: "Origin", favorite: true)
    let empty = LibraryOrganizationSnapshot()
    let first = LibraryOrganizationReconciliation.plan(source: source, destination: empty, previous: nil, hidden: [], newCopy: true)
    #expect(first.result == source)
    let edited = LibraryOrganizationSnapshot(collections: [], tags: ["mine"], title: "Local", favorite: false)
    let next = LibraryOrganizationReconciliation.plan(source: source, destination: edited, previous: first.state, hidden: [])
    #expect(next.result == edited)
    #expect(next.state.suppressedTags == ["draft"])
    #expect(next.state.suppressedCollections == ["work"])
}

@Test func organizationHiddenReassertsAndGenuineSourceReadditionResetsSuppression() {
    let source = LibraryOrganizationSnapshot(collections: ["private"], tags: ["draft"])
    let first = LibraryOrganizationReconciliation.plan(source: source, destination: .init(), previous: nil, hidden: ["private"], newCopy: true)
    let removed = LibraryOrganizationReconciliation.plan(source: source, destination: .init(), previous: first.state, hidden: ["private"])
    #expect(removed.result.collections == ["private"])
    #expect(removed.result.tags.isEmpty)
    let sourceRemoved = LibraryOrganizationReconciliation.plan(source: .init(collections: ["private"]), destination: removed.result, previous: removed.state, hidden: ["private"])
    let readded = LibraryOrganizationReconciliation.plan(source: source, destination: sourceRemoved.result, previous: sourceRemoved.state, hidden: ["private"])
    #expect(readded.result.tags == ["draft"])
}

@Test func organizationLegacyDoesNotOwnExistingMembershipOrResurrectUnknownRemovals() {
    let source = LibraryOrganizationSnapshot(collections: ["work", "private"], tags: ["old"])
    let legacy = LibraryOrganizationReconciliation.plan(source: source, destination: .init(collections: ["work"]), previous: nil, hidden: ["private"])
    #expect(legacy.result.collections == ["work", "private"])
    #expect(legacy.result.tags.isEmpty)
    let removed = LibraryOrganizationReconciliation.plan(source: .init(), destination: legacy.result, previous: legacy.state, hidden: [])
    #expect(removed.result.collections == ["work"])
}

@Test func organizationRemovalKeepsOtherContributorClaims() {
    let source = LibraryOrganizationSnapshot(tags: ["shared", "owned"])
    let first = LibraryOrganizationReconciliation.plan(source: source, destination: .init(), previous: nil, hidden: [], newCopy: true)
    let removed = LibraryOrganizationReconciliation.plan(source: .init(), destination: source, previous: first.state, hidden: [], otherClaims: .init(tags: ["shared"]))
    #expect(removed.result.tags == ["shared"])
}

@Test func localReadditionAfterSuppressionBelongsToDestination() {
    let source = LibraryOrganizationSnapshot(collections: ["draft"], tags: ["work"])
    let first = LibraryOrganizationReconciliation.plan(source: source, destination: .init(), previous: nil, hidden: [], newCopy: true)
    let removed = LibraryOrganizationReconciliation.plan(source: source, destination: .init(), previous: first.state, hidden: [])
    let localReadded = LibraryOrganizationReconciliation.plan(source: source, destination: source, previous: removed.state, hidden: [])
    let sourceRemoved = LibraryOrganizationReconciliation.plan(source: .init(), destination: source, previous: localReadded.state, hidden: [])
    #expect(sourceRemoved.result == source)
}

@Test func scalarOverrideSurvivesLaterSourceClearAndReaddition() {
    let source = LibraryOrganizationSnapshot(title: "Origin", favorite: true)
    let copied = LibraryOrganizationReconciliation.plan(source: source, destination: .init(), previous: nil, hidden: [], newCopy: true)
    let removed = LibraryOrganizationReconciliation.plan(source: source, destination: .init(), previous: copied.state, hidden: [])
    let cleared = LibraryOrganizationReconciliation.plan(source: .init(), destination: .init(), previous: removed.state, hidden: [])
    let readded = LibraryOrganizationReconciliation.plan(source: source, destination: .init(), previous: cleared.state, hidden: [])
    #expect(readded.result.title.isEmpty)
    #expect(!readded.result.favorite)
}

@Test func organizationTagIdentityPreservesDestinationSpellingAndVerifiesServerCase() {
    let source = LibraryOrganizationSnapshot(tags: ["  DRAFT  "])
    let destination = LibraryOrganizationSnapshot(tags: ["Draft"])
    let existing = LibraryOrganizationReconciliation.plan(source: source, destination: destination, previous: nil, hidden: [])
    #expect(existing.result.tags == ["Draft"])
    #expect(existing.state.ownedTags.isEmpty)
    let copied = LibraryOrganizationReconciliation.plan(source: source, destination: .init(), previous: nil, hidden: [], newCopy: true)
    #expect(copied.result == destination)
    #expect(copied.state.ownedTags == ["draft"])
    let removed = LibraryOrganizationReconciliation.plan(source: .init(), destination: .init(tags: ["draft"]), previous: copied.state, hidden: [])
    #expect(removed.result.tags.isEmpty)
}
