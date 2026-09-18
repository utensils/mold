import MoldClient
import SwiftUI

/// Every machine at once: what it is, what it is doing, and whether it is
/// there at all.
///
/// The Machines destination lands HERE, and one machine's page is one push
/// away (`MachinesDestination.swift`). It used to open straight onto whichever
/// machine happened to be selected, which meant the app could show you four
/// GPUs on one box and never once show you the fleet.
struct MachineOverview: View {
    let fleet: MachineFleet
    /// The card the keyboard is on, published outward so the Machine menu
    /// acts on the card you are looking at rather than on a second guess.
    @Binding var focused: MoldHost.ID?
    let perform: (MoldHost.ID, MachineCardActions.Kind) -> Void
    let add: () -> Void

    @FocusState private var focusedCard: MoldHost.ID?
    /// The columns the grid actually drew, so an arrow key means the same
    /// thing the layout does (`MachineGrid.swift`).
    @State private var columns = 1

    var body: some View {
        Group {
            if fleet.cards.isEmpty { empty } else { grid }
        }
        .navigationTitle(title)
        .toolbar { toolbar }
        .onChange(of: focusedCard) { _, card in focused = card }
        .task { await fleet.load() }
    }

    private var title: String {
        let count = fleet.cards.count
        return count == 0 ? "Machines" : "Machines · \(count)"
    }

    private var grid: some View {
        GeometryReader { proxy in
            ScrollView {
                LazyVGrid(columns: items(for: proxy.size.width),
                          spacing: MachineGrid.spacing) {
                    ForEach(fleet.cards) { card in
                        MachineCardView(card: card) { perform(card.id, $0) }
                            .focusable()
                            .focused($focusedCard, equals: card.id)
                            .onMoveCommand { move($0, from: card.id) }
                            .onKeyPress(.return) { perform(card.id, .open); return .handled }
                    }
                }
                .padding(MachineGrid.spacing)
                NearbyMachines(hosts: fleet.hosts, machines: fleet.machines)
                    .padding(.horizontal, MachineGrid.spacing)
                    .padding(.bottom, MachineGrid.spacing)
            }
            .onChange(of: proxy.size.width, initial: true) { _, width in
                columns = MachineGrid.columnCount(for: width)
            }
        }
    }

    /// Equal columns of the count `MachineGrid` decided, rather than
    /// `.adaptive`: the arrow keys need the SAME answer the layout used, and
    /// only a count both can read gives them one.
    private func items(for width: CGFloat) -> [GridItem] {
        Array(repeating: GridItem(.flexible(), spacing: MachineGrid.spacing),
              count: MachineGrid.columnCount(for: width))
    }

    private func move(_ direction: MoveCommandDirection, from card: MoldHost.ID) {
        let step: MachineGrid.Move = switch direction {
        case .up: .up
        case .down: .down
        case .left: .left
        case .right: .right
        @unknown default: .right
        }
        focusedCard = MachineGrid.move(step, from: card, in: fleet.cards, columns: columns)
    }

    private var empty: some View {
        ContentUnavailableView {
            Label("No machines yet", systemImage: "server.rack")
        } description: {
            Text("Add a machine running `mold serve`. Its name or IP is enough.")
        } actions: {
            Button("Add a Machine…", action: add)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem {
            Button(action: add) {
                Label("Add a Machine…", systemImage: "plus")
            }
        }
        ToolbarItem {
            Button { Task { await fleet.refreshAll() } } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .disabled(fleet.cards.isEmpty)
        }
    }
}
