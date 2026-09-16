import SwiftUI

/// Phase 0 placeholder. Each case becomes its own pane as the phases land.
struct DestinationDetail: View {
    let destination: Destination

    var body: some View {
        ContentUnavailableView {
            Label(destination.title, systemImage: destination.symbol)
        } description: {
            Text("Not built yet.")
        }
        .navigationTitle(destination.title)
    }
}
