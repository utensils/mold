import SwiftUI
import WidgetKit

/// The Lock Screen's view of the queue: "2 rendering · 1 held", a ring for
/// the render on screen, and one line inline.
struct QueueStatusWidget: Widget {
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: "QueueStatus", provider: QueueStatusProvider()) { entry in
            QueueStatusView(entry: entry)
                .containerBackground(.clear, for: .widget)
                .widgetURL(DeepLink.queue(job: nil).url)
        }
        .configurationDisplayName("Queue")
        .description("What your machines are rendering and holding.")
        .supportedFamilies([.accessoryRectangular, .accessoryCircular, .accessoryInline])
    }
}

struct QueueStatusProvider: TimelineProvider {
    func placeholder(in context: Context) -> SnapshotEntry { SnapshotEntry(date: .now, snapshot: .empty) }

    func getSnapshot(in context: Context, completion: @escaping (SnapshotEntry) -> Void) {
        completion(SnapshotEntry(date: .now, snapshot: WidgetSnapshot.load()))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<SnapshotEntry>) -> Void) {
        completion(Timeline(entries: [SnapshotEntry(date: .now, snapshot: WidgetSnapshot.load())],
                            policy: .after(.now.addingTimeInterval(30 * 60))))
    }
}

struct QueueStatusView: View {
    @Environment(\.widgetFamily) private var family
    let entry: SnapshotEntry

    var body: some View {
        let snapshot = entry.snapshot
        switch family {
        case .accessoryCircular:
            if let progress = snapshot.progress {
                Gauge(value: progress) {
                    Image(systemName: "wand.and.sparkles").accessibilityLabel("Rendering")
                }
                .gaugeStyle(.accessoryCircularCapacity)
            } else {
                ZStack {
                    AccessoryWidgetBackground()
                    Text("\(snapshot.rendering + snapshot.held)").font(.title3.monospacedDigit())
                }
                .accessibilityLabel(Text(snapshot.queueSummary))
            }
        case .accessoryInline:
            Label(snapshot.queueSummary, systemImage: "list.bullet.indent")
        default:
            VStack(alignment: .leading, spacing: 2) {
                Label("Mold Studio", systemImage: "wand.and.sparkles").font(.headline)
                Text(snapshot.queueSummary)
                if let progress = snapshot.progress {
                    ProgressView(value: progress)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
