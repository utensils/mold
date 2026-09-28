import SwiftUI
import WidgetKit

/// Home Screen widgets and, from M10, the render Live Activity. Reads only the
/// App Group snapshot the app writes: no network, no Keychain (DESIGN.md §5.8).
@main
struct MoldCompanionWidgets: WidgetBundle {
    var body: some Widget {
        RecentPrintsWidget()
    }
}

struct RecentPrintsWidget: Widget {
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: "RecentPrints", provider: RecentPrintsProvider()) { _ in
            RecentPrintsView()
                .containerBackground(.fill.tertiary, for: .widget)
        }
        .configurationDisplayName("Recent Prints")
        .description("Your latest prints from every machine.")
        .supportedFamilies([.systemSmall, .systemMedium])
    }
}

struct RecentPrintsEntry: TimelineEntry {
    let date: Date
}

/// Until the app writes its snapshot (M11) there is nothing to show but the
/// way in, so one entry that never needs refreshing.
struct RecentPrintsProvider: TimelineProvider {
    func placeholder(in context: Context) -> RecentPrintsEntry { RecentPrintsEntry(date: .now) }

    func getSnapshot(in context: Context, completion: @escaping (RecentPrintsEntry) -> Void) {
        completion(RecentPrintsEntry(date: .now))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<RecentPrintsEntry>) -> Void) {
        completion(Timeline(entries: [RecentPrintsEntry(date: .now)], policy: .never))
    }
}

struct RecentPrintsView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.title2)
                .foregroundStyle(.tint)
                .accessibilityHidden(true)
            Spacer(minLength: 0)
            Text("No prints yet")
                .font(.headline)
            Text("Open Mold Studio to add a machine.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    }
}
