import MoldClient
import SwiftUI

/// The Machines destination: the fleet, and one machine's page one push in.
///
/// The destination used to BE a single machine's page, which meant the app
/// could show you four GPUs on one box and never once show you the fleet --
/// the owner's own words: "a top-level view showing all the connected nodes
/// and what their status is".
///
/// The stack's path IS `selectedMachine`, the preference the sidebar already
/// writes (`MachineNavigation`): a machine row opens that machine's page, and
/// Back deselects the row. A second piece of state would be a second answer.
struct MachinesDestination: View {
    // Not `private`: `MachinesDestination+Menu` reads all of these, and
    // `private` does not cross a file boundary even within one type.
    @Environment(HostStore.self) var hosts
    @Environment(MachineStore.self) private var machines
    @Environment(QueueStore.self) private var queue
    @Environment(ModelStore.self) private var models
    @Environment(ActivityStore.self) private var activity
    @Environment(UpscaleStore.self) private var upscales
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) var selectedMachine = ""
    @Binding var destination: Destination

    @State private var isAdding = false
    @State var editing: MoldHost?
    @State var pendingRemoval: Destruction?
    /// The card the keyboard is on, or none. The Machine menu acts on THIS
    /// and nothing else while the overview is up: falling back to the default
    /// machine would make ⌘⌫ remove a machine nobody pointed at.
    @State var focusedCard: MoldHost.ID?
    /// The launch hook is a LAUNCH hook: applying it every time this
    /// destination appears would send you back to the overview each time you
    /// returned from the Library.
    @State private var launchApplied = false

    private var fleet: MachineFleet {
        MachineFleet(hosts: hosts, machines: machines, queue: queue, models: models,
                     activity: activity, upscales: upscales)
    }

    var body: some View {
        NavigationStack(path: path) {
            MachineOverview(fleet: fleet, focused: $focusedCard,
                            perform: perform, add: { isAdding = true })
                .navigationDestination(for: MoldHost.ID.self) { _ in
                    MachinesPane(destination: $destination)
                }
        }
        // The SAME sheet Settings ▸ Machines opens, which is what normalizes
        // the address, checks the machine while you type and refuses one
        // another machine already answers at (`HostEditor.swift`).
        .sheet(isPresented: $isAdding) {
            HostEditor { hosts.add(name: $0, url: $1, apiKey: $2) }
        }
        .sheet(item: $editing) { host in
            HostEditor(host: host) { name, url, key in
                hosts.update(MoldHost(id: host.id, name: name, baseURL: url, apiKey: key))
            }
        }
        .destructionDialog($pendingRemoval)
        // Published ONCE for the whole destination rather than by each half:
        // two views offering the Machine menu a selection is two answers to
        // "which machine", and which one wins is whichever SwiftUI asked last.
        .focusedSceneValue(\.machineSelection, selection)
        .focusedSceneValue(\.refreshAction) { refresh() }
        .task { applyLaunchRequest() }
    }

    private var path: Binding<[MoldHost.ID]> {
        Binding(
            get: { MachineNavigation.path(selected: selectedMachine, in: hosts.hosts) },
            set: { selectedMachine = MachineNavigation.stored(path: $0) }
        )
    }

    /// The machine whose page is open, if the stack is pushed at all. Not
    /// `private`: `MachinesDestination+Menu` reads it, and `private` does not
    /// cross a file boundary even within one type.
    var open: MoldHost? {
        guard let id = path.wrappedValue.last else { return nil }
        return hosts.host(id)
    }

    /// The machine every menu item acts on: the one whose page is open, else
    /// the card the keyboard is on, else none at all.
    var target: MoldHost? {
        open ?? focusedCard.flatMap(hosts.host)
    }

    private func refresh() {
        Task {
            if let open { await fleet.refresh(one: open) } else { await fleet.refreshAll() }
        }
    }

    private func applyLaunchRequest() {
        guard !launchApplied else { return }
        launchApplied = true
        switch MachineLaunch.resolve(destination: NativeUAT.destination.value(),
                                     machine: NativeUAT.machine.value(), in: hosts.hosts) {
        case .unchanged: break
        case .overview: selectedMachine = ""
        case let .machine(id): selectedMachine = id.uuidString
        }
    }
}
