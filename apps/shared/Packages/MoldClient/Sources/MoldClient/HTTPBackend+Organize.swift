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
    func createCollection(name: String, description: String?) async throws -> Collection {
        try await send("/api/gallery/collections", method: "POST",
                       body: CollectionCreate(name: name, description: description))
    }

    /// Absent fields are untouched, so renaming does not clear a cover.
    func updateCollection(id: String, change: CollectionChange) async throws -> Collection {
        try await send("/api/gallery/collections/\(escaped(id))", method: "PATCH", body: change)
    }

    /// Removes the shelf, never its prints — membership is the only thing that
    /// goes away.
    func deleteCollection(id: String) async throws {
        try await delete("/api/gallery/collections/\(escaped(id))")
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
        var request = self.request("/api/gallery/trash")
        request.httpMethod = "DELETE"
        request.timeoutInterval = 300
        _ = try await bytes(for: request)
    }
}

/// This route's body, its own type rather than a dictionary so that a name
/// with a `/` or a `#` in it is JSON-escaped rather than interpolated raw.
/// The COLLECTION bodies moved beside `Collection`: they are `MoldBackend`'s
/// vocabulary, and this one is only ever this PATCH's.
struct TagRename: Encodable, Sendable {
    let name: String
}
