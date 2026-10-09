import MoldClient
import SwiftUI

/// Hidden shelves remain directly accessible; hiding only excludes their
/// prints from the general Library. The same sheet serves phone and iPad.
struct CollectionsSheet: View {
    @Environment(LibraryStore.self) private var library
    @Environment(\.dismiss) private var dismiss
    var machineIDs: Set<MoldHost.ID>? = nil
    let choose: (LibraryScope) -> Void
    @State private var updating: Set<String> = []

    var body: some View {
        NavigationStack {
            List {
                ForEach(library.shelves) { shelf in
                    VStack(alignment: .leading) {
                        Button {
                            choose(.collection(slug: shelf.slug))
                            dismiss()
                        } label: {
                            HStack(alignment: .firstTextBaseline) {
                                Image(systemName: shelf.hidden ? "rectangle.stack.badge.minus" : "rectangle.stack")
                                    .accessibilityHidden(true)
                                Text(shelf.name)
                                Text(library.shelfPresence(shelf, on: machineIDs ?? library.machineIDs) == .unavailable ? "Unavailable" : library.shelfPresence(shelf, on: machineIDs ?? library.machineIDs) == .absent ? "Not on machine" : shelf.count(in: library.scopedPool(on: machineIDs ?? library.machineIDs)).formatted())
                                    .foregroundStyle(.secondaryText)
                                    .font(.body)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .foregroundStyle(.primary)
                            .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel(shelf.name)
                        Toggle("Hide from All Prints", isOn: Binding(
                            get: { shelf.hidden },
                            set: { hidden in
                                updating.insert(shelf.slug)
                                Task {
                                    await library.setShelfHidden(shelf, hidden: hidden)
                                    updating.remove(shelf.slug)
                                }
                            }))
                            .disabled(updating.contains(shelf.slug))
                    }
                }
            }
            .safeAreaInset(edge: .bottom) {
                Text("Hide from All Prints applies to this collection on every machine. Changes retry when machines reconnect.")
                    .font(.footnote).foregroundStyle(.secondaryText).padding().background(.background)
            }
            .accessibilityIdentifier("collections-sheet")
            .navigationTitle("Collections")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}
