import SwiftUI
import WidgetKit

/// Home Screen and Lock Screen widgets and the render Live Activity. Reads
/// only the App Group snapshot the app writes: no network, no Keychain
/// (DESIGN.md §5.7–§5.8).
@main
struct MoldCompanionWidgets: WidgetBundle {
    var body: some Widget {
        RecentPrintsWidget()
        QueueStatusWidget()
        GenerationLiveActivity()
    }
}

/// One timeline entry: the snapshot as the app last wrote it.
struct SnapshotEntry: TimelineEntry {
    let date: Date
    let snapshot: WidgetSnapshot
    var machine: UUID?
    var favouritesOnly = false

    var prints: [WidgetSnapshot.Print] { snapshot.prints(machine: machine, favouritesOnly: favouritesOnly) }
}

/// A print's small JPEG from the App Group, or nothing.
func snapshotImage(_ print: WidgetSnapshot.Print) -> UIImage? {
    UIImage(contentsOfFile: AppGroup.widget.appending(path: print.image).path)
}
