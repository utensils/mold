import MoldClient

/// What one queue row offers right now, on one machine.
///
/// Declared once because four surfaces draw or dispatch the same four
/// actions -- the row's glyph buttons, the row's contextual menu, a batch's
/// group-wide buttons, and the Queue menu's `FocusedValue` -- and they had
/// already drifted apart: Pause was capability-gated in the menu
/// (`QueuePane+Commands.swift`) and ungated on the row, so on a machine that
/// predates per-job pause the button was offered and the request failed.
/// Every gate below is the machine's own answer; none is a state this app
/// guesses at.
struct QueueRowActions: Equatable {
    var pause = false
    var resume = false
    var retry = false
    var cancel = false

    /// Whether this row offers a particular action -- what a group-wide
    /// dispatch asks of each child before sending anything.
    func offers(_ action: QueueRow.Action) -> Bool {
        switch action {
        case .pause: pause
        case .resume: resume
        case .retry: retry
        case .cancel: cancel
        }
    }

    static func resolve(_ entry: QueueEntry, on capabilities: Capabilities?) -> QueueRowActions {
        QueueRowActions(
            // Only a WAITING row. `set_one_queue_job_paused` refuses a
            // running one by name -- "queue job {id} is already running; only
            // waiting jobs can be paused or resumed" (`routes.rs:7706-7710`)
            // -- so offering Pause there was a guaranteed 409. Web reads the
            // same two states (`useQueueInspection.ts:341-348`).
            pause: entry.state == .queued && capabilities?.canPauseOneJob == true,
            resume: entry.state == .paused && capabilities?.canPauseOneJob == true,
            // The host's own answer to "would trying again help". `false`
            // means it needs repair, and a Retry there would just hold the
            // job again (`routes.rs:7651-7655`).
            retry: entry.state == .held && entry.retryable != false,
            cancel: cancels(entry, on: capabilities))
    }

    /// Every live row is cancellable EXCEPT a running one on a machine that
    /// does not advertise cooperative cancellation -- there being nothing
    /// there to stop work at a safe point. See `canCancelRunningJob`.
    private static func cancels(_ entry: QueueEntry, on capabilities: Capabilities?) -> Bool {
        guard entry.state.isLive else { return false }
        guard entry.state == .running else { return true }
        return capabilities?.canCancelRunningJob == true
    }

    /// A whole batch's row: what it offers is what ANY of its children do.
    /// The group dispatch then asks each child again, so a batch with one
    /// waiting and one running child pauses the waiting one and leaves the
    /// other alone rather than refusing both.
    static func group(_ rows: [QueueEntry], on capabilities: Capabilities?) -> QueueRowActions {
        rows.reduce(into: QueueRowActions()) { union, entry in
            let row = resolve(entry, on: capabilities)
            union.pause = union.pause || row.pause
            union.resume = union.resume || row.resume
            union.retry = union.retry || row.retry
            union.cancel = union.cancel || row.cancel
        }
    }

    /// Everything a row's menu can offer. Moving is here too, because a menu
    /// is the only way to reach a move from the keyboard.
    enum Kind: Hashable {
        case pause, resume, retry, moveUp, moveDown, cancel
    }

    /// THE list the contextual menu draws -- rendered by `.rowActionMenu`,
    /// not mirrored by it. It used to be a `menuTitles` beside a hand-written
    /// `@ViewBuilder`, which is two lists: reordering the view broke nothing,
    /// and "destructive last, behind a divider" was pinned nowhere at all.
    ///
    /// Deliberately the same words and the same order as the Queue menu
    /// itself (`QueueSelection.offeredTitles`), which a test asserts rather
    /// than a comment claiming it.
    func offered(canMoveUp: Bool = false, canMoveDown: Bool = false) -> [RowAction<Kind>] {
        var items: [RowAction<Kind>] = []
        if pause { items.append(RowAction(kind: .pause, title: "Pause Job")) }
        if resume { items.append(RowAction(kind: .resume, title: "Resume Job")) }
        if retry { items.append(RowAction(kind: .retry, title: "Try Again")) }
        if canMoveUp { items.append(RowAction(kind: .moveUp, title: "Move Up")) }
        if canMoveDown { items.append(RowAction(kind: .moveDown, title: "Move Down")) }
        if cancel { items.append(RowAction(kind: .cancel, title: "Cancel Job", isDestructive: true)) }
        return RowAction.ordered(items)
    }

    /// A whole batch's menu. The same gates and the same order; the words say
    /// what each item REACHES, which is what distinguishes a group item from
    /// a child row's own.
    func groupOffered(canMoveUp: Bool = false, canMoveDown: Bool = false) -> [RowAction<Kind>] {
        var items: [RowAction<Kind>] = []
        if pause { items.append(RowAction(kind: .pause, title: "Pause Every Job")) }
        if resume { items.append(RowAction(kind: .resume, title: "Resume Every Job")) }
        if canMoveUp { items.append(RowAction(kind: .moveUp, title: "Move Up")) }
        if canMoveDown { items.append(RowAction(kind: .moveDown, title: "Move Down")) }
        if cancel {
            items.append(RowAction(kind: .cancel, title: "Cancel Every Job", isDestructive: true))
        }
        return RowAction.ordered(items)
    }
}
