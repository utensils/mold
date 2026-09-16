import SwiftUI

/// Routes the sidebar's selection to its pane.
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
            QueuePane()
        }
    }
}
