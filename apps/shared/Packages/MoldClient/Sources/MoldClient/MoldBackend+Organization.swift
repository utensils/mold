import Foundation

/// Collections, tags, and the trash's whole-shelf operations.
public protocol MoldOrganizationBackend: Sendable {
    func collections() async throws -> [Collection]
    func createCollection(name: String, description: String?) async throws -> Collection
    /// Absent fields are untouched, so renaming does not clear a cover.
    func updateCollection(id: String, change: CollectionChange) async throws -> Collection
    /// Removes the shelf, never its prints.
    func deleteCollection(id: String) async throws
    func tags() async throws -> [TagCount]
    @discardableResult
    func renameTag(_ name: String, to newName: String) async throws -> TagCount
    func deleteTag(_ name: String) async throws
    /// Empties the trash now. Permanent.
    func emptyTrash() async throws
}
