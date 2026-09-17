import MoldClient
import SwiftUI

/// Where the next render goes (M8 design, decision 2): Auto, or a machine
/// pinned explicitly.
///
/// The pure menu contents -- `Row`, `Rows`, `rows`, `model(after:)` -- live
/// in `MachineControl+Rows.swift`, which extends this struct rather than
/// declaring its own type.
struct MachineControl: View {
    @Environment(HostStore.self) private var hosts
    @Environment(ModelStore.self) private var models
    @Environment(GenerateController.self) private var controller

    var body: some View {
        let rows = MachineControl.rows(
            hosts: hosts.hosts, chosen: controller.machineChoice, preferred: hosts.preferredHost,
            isUp: hosts.isUp, generates: { hosts.capabilities(of: $0)?.generates == true }
        )
        Menu {
            Button { choose(nil) } label: {
                if rows.chosen == nil {
                    Label("Auto", systemImage: "checkmark")
                } else {
                    Text("Auto")
                }
            }
            .help("Whichever machine is set as default, or the first one that's up")
            Divider()
            ForEach(rows.machines) { row in
                Button { choose(row.id) } label: {
                    if row.id == rows.chosen {
                        Label(row.caption, systemImage: "checkmark")
                    } else {
                        Text(row.caption)
                    }
                }
            }
        } label: {
            Text(rows.label)
        }
        .menuStyle(.button)
        .buttonStyle(.accessoryBar)
        .fixedSize()
        .help("Where this render runs")
    }

    /// Sets the choice, then re-resolves the model on the machine it now
    /// points at -- the same name when it's ready there, else that machine's
    /// first ready model, else none at all (M8 decision 2).
    private func choose(_ id: MoldHost.ID?) {
        controller.machineChoice = id
        guard let machine = id.flatMap(hosts.host) ?? hosts.preferredHost else { return }
        resolveModel(on: machine)
        guard !models.hasLoaded(on: machine.id) else { return }
        Task {
            await models.refresh(on: machine.id)
            // A second press may have moved the choice elsewhere while this
            // refresh was in flight -- re-resolving against a machine that
            // is no longer chosen would silently steal the model field back.
            guard controller.machineChoice == id else { return }
            resolveModel(on: machine)
        }
    }

    private func resolveModel(on machine: MoldHost) {
        let picked = MachineControl.model(after: controller.modelName, readyThere: models.ready(on: machine.id))
        if let picked {
            controller.select(model: picked, on: machine.id)
        } else {
            controller.modelName = nil
        }
    }
}
