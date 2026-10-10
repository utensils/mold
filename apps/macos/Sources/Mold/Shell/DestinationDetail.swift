import SwiftUI
import MoldClient

/// Routes the sidebar's selection to its pane.
struct DestinationDetail: View {
    @Environment(LibraryNavigation.self) private var navigation
    @Binding var destination: Destination

    var body: some View {
        Group { switch destination {
        case .generate:
            GeneratePane(destination: $destination)
        case .library:
            LibraryPane(destination: $destination)
        case .sourceMedia:
            SourceMediaBrowser()
        case .models:
            ModelsPane()
        case .queue:
            QueuePane()
        case .machines:
            // The fleet, with one machine's page a push inside it -- never
            // `MachinesPane` alone, which IS that page.
            MachinesDestination(destination: $destination)
        } }
        .environment(\.sourceMediaStaged, { destination = .generate })
        .environment(\.sourceMediaOwner, { id in
            navigation.scope = .all
            navigation.query = LibraryQuery()
            navigation.reveal = id
            destination = .library
        })
    }
}
