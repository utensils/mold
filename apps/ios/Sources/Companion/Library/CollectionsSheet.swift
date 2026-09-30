import MoldClient
import SwiftUI

/// Hidden shelves remain directly accessible; hiding only excludes their
/// prints from the general Library. The same sheet serves phone and iPad.
struct CollectionsSheet: View {
    @Environment(LibraryStore.self) private var library
    @Environment(\.dismiss) private var dismiss
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
                            Label(shelf.name, systemImage: shelf.hidden ? "rectangle.stack.badge.minus" : "rectangle.stack")
                        }
                        .buttonStyle(.borderless)
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
            .navigationTitle("Collections")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}
