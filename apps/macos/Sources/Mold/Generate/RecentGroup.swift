import MoldClient
import MoldStyle
import SwiftUI

/// What a machine remembers being asked for, offered back as a starting
/// point -- never as a full restore.
///
/// Picking a row changes only the prompt: the row's own model may not be
/// installed on the machine currently chosen, may not be the one picked, and
/// adopting it would silently reset every control the recipe owns. Per
/// machine and never merged, because a prompt typed against one machine is
/// that machine's.
struct RecentGroup: View {
    let host: MoldHost?
    @Binding var draft: RenderDraft
    @Binding var isExpanded: Bool
    /// Whether a render is in flight. A settle (busy -> not busy) is one of
    /// the two moments this section refreshes; the other is the toolbar
    /// button and first expansion, both below.
    let isBusy: Bool

    @Environment(PromptHistoryStore.self) private var history
    @Environment(HostStore.self) private var hosts
    @State private var pendingDestruction: LibraryActions.Destruction?

    var body: some View {
        // The same section every other group is, so its rows lead from the
        // title's edge too -- the Refresh rides the title as an accessory
        // rather than as a second thing in the content.
        InspectorSection(title, isExpanded: $isExpanded) {
            Button { Task { await refresh() } } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.borderless)
            .help("Refresh")
        } content: {
            content
        }
        .task(id: taskKey) { if isExpanded { await refreshIfNeeded() } }
        .onChange(of: isBusy) { wasBusy, nowBusy in
            guard wasBusy, !nowBusy else { return }
            Task { await refresh() }
        }
        .destructionDialog($pendingDestruction)
    }

    @ViewBuilder private var content: some View {
        if let host {
            switch Listing.resolve(
                hasLoaded: history.hasLoaded(on: host.id),
                isUnavailable: history.unavailable.contains(host.id),
                entries: history.entries(on: host.id)
            ) {
            case .loading:
                ProgressView()
            case .unavailable:
                Text("This machine doesn't keep a prompt history.")
                    .font(.caption).foregroundStyle(.secondary)
            case .empty:
                Text("Prompts you send appear here.")
                    .font(.caption).foregroundStyle(.secondary)
            case let .rows(entries):
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(entries) { entry in
                        row(entry)
                        if entry.id != entries.last?.id { Divider() }
                    }
                    Button("Clear Recent…", role: .destructive) { confirmClear(on: host) }
                        .buttonStyle(.plain)
                        .foregroundStyle(.red)
                }
            }
        }
    }

    private var title: String {
        guard let host else { return "Recent" }
        return hosts.hosts.count > 1 ? "Recent on \(host.name)" : "Recent"
    }

    private var taskKey: String { "\(host?.id.uuidString ?? "")-\(isExpanded)" }

    private func refresh() async {
        guard let host else { return }
        await history.refresh(on: host.id)
    }

    private func refreshIfNeeded() async {
        guard let host, !history.hasLoaded(on: host.id) else { return }
        await history.refresh(on: host.id)
    }

    private func confirmClear(on host: MoldHost) {
        pendingDestruction = Self.clearDestruction(machine: host.name) {
            Task { await history.clear(on: host.id) }
        }
    }
}

extension RecentGroup {
    /// What the section says, resolved once from the store's own three
    /// answers rather than the view guessing at their combination.
    enum Listing: Equatable {
        case loading
        case unavailable
        case empty
        case rows([HistoryEntry])

        static func resolve(hasLoaded: Bool, isUnavailable: Bool, entries: [HistoryEntry]) -> Listing {
            guard hasLoaded else { return .loading }
            if isUnavailable { return .unavailable }
            return entries.isEmpty ? .empty : .rows(entries)
        }
    }

    /// Puts the prompt back -- and only the prompt.
    static func pick(_ entry: HistoryEntry, into draft: inout RenderDraft) {
        draft.prompt = entry.prompt
    }

    /// There is no per-row delete: the server has no route for one, only a
    /// whole-history clear.
    static func clearDestruction(machine: String, clear: @escaping () -> Void) -> LibraryActions.Destruction {
        LibraryActions.Destruction(
            title: "Clear the prompt history on \(machine)?",
            message: "Every prompt this machine remembers is removed. The prints themselves are not touched.",
            verb: "Clear",
            perform: clear
        )
    }
}
