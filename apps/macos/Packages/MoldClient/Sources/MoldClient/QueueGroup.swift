import Foundation

/// One machine's queue as the pane draws it: a flat row, or a batch and its
/// children.
///
/// Children order by `revision`, which is THE authority, and fall back to
/// `updatedAtMs` only when both sides are 0 or absent -- retry is the one
/// transition that moves a child BACKWARD (held -> accepted) and
/// same-millisecond commits are routine, so a timestamp comparison can hide a
/// retry outright. `BatchChild.supersedes(_:)` already writes this rule down
/// (`BatchStatus.swift:37-42`) and is reused, not restated -- `build` uses it
/// itself to resolve duplicate views of one child before trusting any of them
/// for order, rather than assuming a caller already deduped.
public struct QueueGroup: Identifiable, Hashable, Sendable {
    public let batchId: String?
    public let rows: [QueueEntry]

    /// A batch of one is a plain row: a disclosure triangle over a single
    /// child is a control that reveals nothing.
    public var isExpandable: Bool { rows.count > 1 }

    public var id: String { batchId ?? rows[0].id }

    /// The one entry a `List` selection names, or nil -- the selection is a
    /// GROUP id, and a batch of one is drawn as a plain row under its BATCH
    /// id, so looking the selection up among entry ids found nothing for
    /// every render this app submits (each is a batch of one): the Queue
    /// menu offered no Pause or Cancel for a row whose own contextual menu
    /// did (UAT 2026-09-17 #6). An expandable group's own row resolves to no
    /// entry, which is correct: none of the menu's items are batch-wide.
    public static func selectedEntry(_ selection: String, in groups: [QueueGroup]) -> QueueEntry? {
        guard let group = groups.first(where: { $0.id == selection }), !group.isExpandable
        else { return nil }
        return group.rows[0]
    }

    /// Groups by `batchId`, keeps each group where its FIRST row sat in the
    /// listing (so grouping never reorders the machine's queue), and orders
    /// within a group by the batch's own children when they are known and by
    /// `batchIndex` when they are not.
    public static func build(
        _ entries: [QueueEntry], children: [String: [BatchChild]]
    ) -> [QueueGroup] {
        var order: [String] = []
        var rowsByKey: [String: [QueueEntry]] = [:]
        for entry in entries {
            let key = entry.batchId ?? entry.id
            if rowsByKey[key] == nil {
                order.append(key)
                rowsByKey[key] = []
            }
            rowsByKey[key]?.append(entry)
        }
        return order.map { key in
            let rows = rowsByKey[key] ?? []
            let batchId = rows.first?.batchId
            let known = batchId.flatMap { children[$0] }
            return QueueGroup(batchId: batchId, rows: ordered(rows, by: known))
        }
    }

    /// Sorts a group's rows by its resolved children's `index` when the batch
    /// is known, and by the row's own `batchIndex` otherwise. `known` may
    /// carry more than one view of the same child -- an event-driven update
    /// arriving out of order, say -- so the newest view per job id is
    /// resolved via `supersedes` before its `index` is trusted for order.
    private static func ordered(_ rows: [QueueEntry], by known: [BatchChild]?) -> [QueueEntry] {
        guard let known, !known.isEmpty else {
            return rows.sorted { ($0.batchIndex ?? .max) < ($1.batchIndex ?? .max) }
        }
        var resolved: [String: BatchChild] = [:]
        for child in known where resolved[child.jobId].map({ child.supersedes($0) }) ?? true {
            resolved[child.jobId] = child
        }
        return rows.sorted {
            let lhs = resolved[$0.id]?.index ?? $0.batchIndex ?? .max
            let rhs = resolved[$1.id]?.index ?? $1.batchIndex ?? .max
            return lhs < rhs
        }
    }
}
