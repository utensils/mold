import Foundation
import MoldClient

/// One machine, as the fleet overview draws it.
///
/// A pure value, built from the stores the machine's own page already reads --
/// so what a card says can be asked without a window, and the view is a
/// renderer with no opinions of its own. Every figure is OPTIONAL and an
/// absent one draws NO row: a card is read at a glance, and a column of em
/// dashes is a column of nothing.
struct MachineCard: Identifiable, Equatable {
    let id: MoldHost.ID
    let name: String
    /// What the sidebar and Settings print for this machine's address.
    let address: String
    /// This Mac's own engine. It is not a saved row, so there is nothing to
    /// edit, nothing to remove and nothing to pair (`MoldEngine.isPairable`).
    let isThisMac: Bool
    let isDefault: Bool
    let reachability: HostStore.Reachability
    /// The GPUs said the way a person says them: "4× NVIDIA L40S".
    let gpus: String?
    /// Every card's load averaged into one figure. Absent unless at least one
    /// card reports utilization at all -- only NVML does.
    let gpuLoad: String?
    let gpuMemory: MemoryFigure?
    let systemMemory: MemoryFigure?
    /// `MachineFigures`' own sentences, or nothing at all: `nil` here is "the
    /// store has not answered for this machine yet", which is not a fact worth
    /// a row on a card.
    let work: String?
    let models: String?

    /// A used-of-total figure and the sentence beside it, so the view draws a
    /// bar only where there is something true to draw (`MemoryBar.swift`).
    struct MemoryFigure: Equatable {
        let reading: MemoryReading
        let text: String
    }

    /// The short form the sidebar row prints -- "Ready · 0.29.0", "Busy ·
    /// 0.29.0", "Checking…", "Needs an API key", or the machine's own reason.
    var status: String? { reachability.summary }

    /// The one line under the name when the machine is not answering
    /// normally. `HostStatus`'s long form, which says WHAT answered rather
    /// than only that something did; absent on a machine that is up, whose
    /// figures speak for themselves.
    var reason: String? {
        if case .up = reachability { return nil }
        return reachability.sentence
    }

    /// The reason, unless it is word for word what the status line already
    /// says. A machine that is down reads its own sentence in BOTH forms, and
    /// a card printing it twice is a card nobody reads once.
    var explanation: String? {
        guard let reason, reason != status else { return nil }
        return reason
    }

    /// Down keeps its place in the grid, dimmed. Moving a machine because it
    /// stopped answering is how you lose the one you were looking for.
    var isDimmed: Bool {
        if case .down = reachability { return true }
        return false
    }
}

extension MachineCard {
    /// Built from exactly what the machine's own page reads: `HostStore` for
    /// reachability, `MachineStore` for devices and the 1 Hz snapshot,
    /// `QueueStore` and `ModelStore` for the two counts. Nothing here fetches.
    ///
    /// `live` and `models` are `nil` until their store has answered for this
    /// machine -- the distinction `MachineFigures` exists to keep.
    init(host: MoldHost, reachability: HostStore.Reachability, isDefault: Bool,
         devices: [DeviceInfo], snapshot: ResourceSnapshot?,
         live: [QueueEntry]?, alsoRunning: Int = 0, models: [Model]?) {
        id = host.id
        name = host.name
        address = HostAddress.displayString(for: host.baseURL)
        isThisMac = host.id == MoldEngine.localHostID
        self.isDefault = isDefault
        self.reachability = reachability

        // A machine that is DOWN shows what it is and why, and no figures:
        // its last-known GPU rows are kept by `MachineStore` on purpose, but
        // printing them beside a red dot says they are current. Its own page
        // makes the same choice (`MachinesPane.unreachable`). Checking is a
        // machine we know being asked again, so its figures stand.
        guard MachineCard.figuresStand(under: reachability) else {
            gpus = nil
            gpuLoad = nil
            gpuMemory = nil
            systemMemory = nil
            work = nil
            self.models = nil
            return
        }
        var status: ServerStatus?
        if case let .up(answered) = reachability { status = answered }
        gpus = MachineCardFigures.gpus(devices: devices, reported: status?.gpus)
        gpuLoad = MachineCardFigures.load(devices: devices, snapshot: snapshot)
        gpuMemory = MachineCardFigures.videoMemory(devices: devices, snapshot: snapshot)
        systemMemory = MachineCardFigures.systemMemory(snapshot)
        work = live.map { MachineFigures.workFigure(live: $0, alsoRunning: alsoRunning) }
        self.models = models.map { MachineFigures.modelFigure(ready: $0) }
    }

    /// Whether the figures a card would print are still about the machine in
    /// front of you.
    static func figuresStand(under reachability: HostStore.Reachability) -> Bool {
        switch reachability {
        case .up, .checking: true
        // Not "no figures yet" but "no figures from us": a machine refusing
        // this app's key has told us nothing we may print.
        case .down, .needsKey, .unknown: false
        }
    }

    /// The default machine first, then by name.
    ///
    /// Work goes to the default unless something else says so
    /// (`HostStore+Default`), which makes it the one card a person looks for
    /// first. Everything after it is alphabetical rather than in the order
    /// machines happened to be added -- and a case-insensitive, numeral-aware
    /// comparison, so "GPU 10" sorts after "GPU 9".
    static func sorted(_ cards: [MachineCard]) -> [MachineCard] {
        cards.sorted { left, right in
            if left.isDefault != right.isDefault { return left.isDefault }
            return left.name.localizedStandardCompare(right.name) == .orderedAscending
        }
    }
}
