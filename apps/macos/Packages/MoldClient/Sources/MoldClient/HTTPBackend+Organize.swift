import Foundation

/// Collections, tags and the trash — what a person does to a library after the
/// pictures exist.
///
/// Every call here names ONE machine, because a collection's id and a tag's
/// membership belong to that machine. Folding several machines' answers into
/// the one shelf a person sees is `CollectionShelf.merge`, never a request.
public extension HTTPBackend {

    // MARK: - Collections

    /// Makes a collection and hands back the machine's own row, so the caller
    /// learns the id and the slug it resolved to rather than guessing either.
    func createCollection(name: String, description: String? = nil) async throws -> Collection {
        try await send("/api/gallery/collections", method: "POST",
                       body: CollectionCreate(name: name, description: description))
    }

    /// Absent fields are untouched, so renaming does not clear a cover.
    func updateCollection(id: String, change: CollectionChange) async throws -> Collection {
        try await send("/api/gallery/collections/\(id)", method: "PATCH", body: change)
    }

    /// Removes the shelf, never its prints — membership is the only thing that
    /// goes away.
    func deleteCollection(id: String) async throws {
        try await delete("/api/gallery/collections/\(id)")
    }

    /// One call adds and removes, which is what makes dragging between two
    /// shelves a single request rather than a pair that can half-fail.
    @discardableResult
    func changeCollectionItems(id: String, add: [String] = [],
                               remove: [String] = []) async throws -> Collection {
        try await send("/api/gallery/collections/\(id)/items", method: "PUT",
                       body: CollectionItems(add: add, remove: remove))
    }

    // MARK: - Tags

    /// Renaming rewrites the tag on every print carrying it, which is why it
    /// is one request and not a loop the app could fail halfway through.
    @discardableResult
    func renameTag(_ name: String, to newName: String) async throws -> TagCount {
        try await send("/api/gallery/tags/\(escaped(name))", method: "PATCH",
                       body: TagRename(name: newName))
    }

    func deleteTag(_ name: String) async throws {
        try await delete("/api/gallery/tags/\(escaped(name))")
    }

    // MARK: - Trash

    /// Empties the trash now. Permanent, and the only call here that destroys
    /// anything.
    func emptyTrash() async throws {
        try await delete("/api/gallery/trash")
    }

    /// Purges only what is already past its retention — what the host would
    /// have done on its own schedule anyway.
    func sweepTrash() async throws {
        _ = try await postRaw("/api/gallery/trash/sweep", body: Empty())
    }
}

/// Bodies. Each is its own type rather than a dictionary so that an optional
/// left nil is OMITTED: on a PATCH, absent means "untouched" and null would
/// mean "clear it".
public struct CollectionCreate: Encodable, Sendable {
    public let name: String
    public let description: String?
}

public struct CollectionChange: Encodable, Sendable {
    public var name: String?
    public var coverFilename: String?
    public var hidden: Bool?

    public init(name: String? = nil, coverFilename: String? = nil, hidden: Bool? = nil) {
        self.name = name
        self.coverFilename = coverFilename
        self.hidden = hidden
    }
}

struct CollectionItems: Encodable, Sendable {
    let add: [String]
    let remove: [String]
}

struct TagRename: Encodable, Sendable {
    let name: String
}

struct Empty: Encodable, Sendable {}
