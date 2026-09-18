import SwiftUI

/// Routes the sidebar's selection to its pane.
struct DestinationDetail: View {
    @Binding var destination: Destination

    var body: some View {
        switch destination {
        case .generate:
            GeneratePane(destination: $destination)
        case .library:
            LibraryPane(destination: $destination)
        case .models:
            ModelsPane()
        case .queue:
            QueuePane()
        case .machines:
            // The fleet, with one machine's page a push inside it -- never
            // `MachinesPane` alone, which IS that page.
            MachinesDestination(destination: $destination)
        }
    }
}
