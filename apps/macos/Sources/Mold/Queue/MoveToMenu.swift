import MoldClient
import SwiftUI

/// "Move to ▾" on a held row: every machine that is up and generates,
/// captioned with its queue depth. Absent, not disabled, when the list is
/// empty (decision 19) -- a control whose only outcome is "there's nowhere
/// to send it" is worse than no control at all.
struct MoveToMenu: View {
    let destinations: [TransferStore.TransferDestination]
    let send: (MoldHost.ID) -> Void

    var body: some View {
        if !destinations.isEmpty {
            Menu("Move to") {
                ForEach(destinations) { destination in
                    Button(destination.caption) { send(destination.id) }
                }
            }
        }
    }
}
