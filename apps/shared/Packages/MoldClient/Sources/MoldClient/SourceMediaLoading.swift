import Foundation

/// Shared publication fencing for source reads. Each check observes the live
/// route, instance, content revision and visibility supplied by the owning app.
@MainActor
public enum SourceMediaLoading {
    public static func read<Value>(isCurrent: () -> Bool, operation: () async throws -> Value) async throws -> Value? {
        guard !Task.isCancelled, isCurrent() else { return nil }
        do {
            let value = try await operation()
            guard !Task.isCancelled, isCurrent() else { return nil }
            return value
        } catch {
            guard !Task.isCancelled, isCurrent() else { return nil }
            throw error
        }
    }

    public struct Original {
        public let file: SourceMediaFile
        public let inspection: SourceMediaOriginal.Inspection
    }

    public static func original(row: SourceMediaCatalog.Row, backend: any MoldBackend,
                                isCurrent: () -> Bool) async throws -> Original? {
        guard let file = try await read(isCurrent: isCurrent, operation: {
            try await backend.retainedSourceMediaFile(for: row.owner.print.filename, member: row.member)
        }), let inspection = try await read(isCurrent: isCurrent, operation: {
            try await SourceMediaOriginal.inspect(file)
        }) else { return nil }
        return Original(file: file, inspection: inspection)
    }

    public static func inventory(owner: LibraryEntry, authority: SourceMediaAuthority, catalog: SourceMediaCatalog,
                                 backend: any MoldBackend, refresh: Bool, isCurrent: () -> Bool) async throws -> RetainedSourceMedia.Inventory? {
        guard let route = authority.route else { return nil }
        return try await read(isCurrent: isCurrent) {
            try await catalog.inventory(for: owner, backend: backend, route: route,
                                        instance: authority.instance, refresh: refresh)
        }
    }
}

public struct SourceMediaOriginalKey: Hashable, Sendable {
    public let row: SourceMediaCatalog.Row?
    public let revision: Int
    public let retry: Int
    public init(row: SourceMediaCatalog.Row?, revision: Int, retry: Int) {
        self.row = row; self.revision = revision; self.retry = retry
    }
}

public struct SourceMediaInventoryKey: Hashable, Sendable {
    public let authority: SourceMediaAuthority
    public let retry: Int
    public let revision: Int
    public init(authority: SourceMediaAuthority, retry: Int, revision: Int) {
        self.authority = authority; self.retry = retry; self.revision = revision
    }
}
