import SwiftUI

/// Routes the sidebar's selection to its pane. Destinations not yet built say
/// so plainly rather than showing an empty screen.
struct DestinationDetail: View {
    let destination: Destination

    var body: some View {
        switch destination {
        case .generate:
            GeneratePane()
        case .library:
            LibraryPane()
        case .models:
            ModelsPane()
        case .queue:
            ContentUnavailableView {
                Label(destination.title, systemImage: destination.symbol)
            } description: {
                Text("Not built yet.")
            }
            .navigationTitle(destination.title)
        }
    }
}
