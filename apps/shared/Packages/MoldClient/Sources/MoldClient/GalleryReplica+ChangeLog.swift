import Foundation

/// A mirror's position in one replica: the epoch (which replica object) and
/// the revision it last planned from. Persisted by the mirror, never by the
/// replica, so a relaunch -- a new epoch -- is always a full plan.
public struct ReplicaCursor: Codable, Sendable, Equatable {
    public let epoch: UUID
    public let revision: Int

    public init(epoch: UUID, revision: Int) {
        self.epoch = epoch
        self.revision = revision
    }
}

/// What a mirror plans from: the replica's rows at one revision, read in one
/// step so the cursor it records names exactly the state it read.
///
/// Copy-on-write: taking a view copies nothing, and nothing here is filtered
/// up front -- a mirror that only looks up the names that changed pays for
/// those names.
public struct ReplicaMirrorView: Sendable {
    public let cursor: ReplicaCursor
    /// Every row, live and trashed, by filename.
    public let rows: [String: GalleryPrint]
    /// Names the legacy trash listing put in Recently Deleted.
    let listedInTrash: Set<String>

    public func isTrash(_ name: String) -> Bool {
        listedInTrash.contains(name) || rows[name]?.trashedAt != nil
    }

    /// The live row for `name`, or nil when it is trashed or absent.
    public func live(_ name: String) -> GalleryPrint? {
        guard let row = rows[name], !isTrash(name) else { return nil }
        return row
    }
}

// The bounded change log behind a mirror's cursor.
extension GalleryReplica {
    /// Records the names one delivered revision touched.
    func logChange(_ names: some Swift.Collection<String>) {
        changeLog.append((revision, Array(names)))
        changeLogNames += names.count
        while changeLogNames > changeLogLimit, let oldest = changeLog.first {
            changeLog.removeFirst()
            changeLogNames -= oldest.names.count
            changeLogFloor = oldest.revision
        }
    }

    /// Every name a delivered revision after `cursor` touched, or nil when
    /// the log cannot say: another epoch, a revision this replica never
    /// reached, or one older than what the bounded log still holds. Nil means
    /// "plan everything", never "nothing changed".
    public func changedNames(since cursor: ReplicaCursor) -> Set<String>? {
        guard cursor.epoch == epoch, cursor.revision >= changeLogFloor, cursor.revision <= revision
        else { return nil }
        var names = Set<String>()
        for entry in changeLog.reversed() {
            guard entry.revision > cursor.revision else { break }
            names.formUnion(entry.names)
        }
        return names
    }

    /// The rows now, with the cursor that names them.
    public func mirrorView() -> ReplicaMirrorView {
        ensureLoaded()
        return ReplicaMirrorView(cursor: ReplicaCursor(epoch: epoch, revision: revision),
                                 rows: rows, listedInTrash: listedInTrash)
    }
}
