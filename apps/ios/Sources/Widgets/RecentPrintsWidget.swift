import AppIntents
import SwiftUI
import WidgetKit

/// Which machine's prints, and whether only favourites.
struct RecentPrintsConfiguration: WidgetConfigurationIntent {
    static let title: LocalizedStringResource = "Recent Prints"
    static let description = IntentDescription("Your latest prints, from every machine or one.")

    @Parameter(title: "Machine") var machine: MachineEntity?
    @Parameter(title: "Favourites Only", default: false) var favouritesOnly: Bool
}

/// A machine by name, from the snapshot (the widget never reads hosts.json's
/// addresses, only what the app chose to show it).
struct MachineEntity: AppEntity {
    static let typeDisplayRepresentation: TypeDisplayRepresentation = "Machine"
    static let defaultQuery = MachineQuery()
    let id: UUID
    let name: String
    var displayRepresentation: DisplayRepresentation { DisplayRepresentation(title: "\(name)") }
}

struct MachineQuery: EntityQuery {
    func entities(for identifiers: [UUID]) async throws -> [MachineEntity] {
        try await suggestedEntities().filter { identifiers.contains($0.id) }
    }

    func suggestedEntities() async throws -> [MachineEntity] {
        WidgetSnapshot.load().machines.map { MachineEntity(id: $0.id, name: $0.name) }
    }
}

struct RecentPrintsWidget: Widget {
    var body: some WidgetConfiguration {
        AppIntentConfiguration(kind: "RecentPrints", intent: RecentPrintsConfiguration.self,
                               provider: RecentPrintsProvider()) { entry in
            RecentPrintsView(entry: entry)
                .containerBackground(.fill.tertiary, for: .widget)
        }
        .configurationDisplayName("Recent Prints")
        .description("Your latest prints from every machine.")
        .supportedFamilies([.systemSmall, .systemMedium, .systemLarge])
        .contentMarginsDisabled()
    }
}

struct RecentPrintsProvider: AppIntentTimelineProvider {
    func placeholder(in context: Context) -> SnapshotEntry { SnapshotEntry(date: .now, snapshot: .empty) }

    func snapshot(for configuration: RecentPrintsConfiguration, in context: Context) async -> SnapshotEntry {
        entry(configuration)
    }

    /// The app reloads timelines whenever the snapshot changes; this is only
    /// the fallback.
    func timeline(for configuration: RecentPrintsConfiguration, in context: Context) async -> Timeline<SnapshotEntry> {
        Timeline(entries: [entry(configuration)], policy: .after(.now.addingTimeInterval(60 * 60)))
    }

    private func entry(_ configuration: RecentPrintsConfiguration) -> SnapshotEntry {
        SnapshotEntry(date: .now, snapshot: WidgetSnapshot.load(), machine: configuration.machine?.id,
                      favouritesOnly: configuration.favouritesOnly)
    }
}

struct RecentPrintsView: View {
    @Environment(\.widgetFamily) private var family
    let entry: SnapshotEntry

    var body: some View {
        let prints = entry.prints
        if prints.isEmpty {
            EmptyPrints(hasMachines: !entry.snapshot.machines.isEmpty)
                .padding()
        } else {
            switch family {
            case .systemSmall:
                PrintCell(print: prints[0], showsTitle: true)
                    .widgetURL(prints[0].link.url)
            case .systemMedium:
                grid(Array(prints.prefix(4)), columns: 4)
            default:
                grid(Array(prints.prefix(9)), columns: 3)
            }
        }
    }

    private func grid(_ prints: [WidgetSnapshot.Print], columns: Int) -> some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 4), count: columns), spacing: 4) {
            ForEach(prints) { print in
                Link(destination: print.link.url) {
                    PrintCell(print: print, showsTitle: false).aspectRatio(1, contentMode: .fit)
                }
            }
        }
        .padding(8)
    }
}

private struct PrintCell: View {
    let print: WidgetSnapshot.Print
    let showsTitle: Bool

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            if let image = snapshotImage(print) {
                Image(uiImage: image).resizable().widgetAccentedRenderingMode(.fullColor).scaledToFill()
            } else {
                Rectangle().fill(.quaternary)
            }
            if showsTitle {
                Text(print.title)
                    .font(.caption.weight(.semibold))
                    .lineLimit(2)
                    .foregroundStyle(.white)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.black.opacity(0.55))
            }
        }
        .clipShape(.rect(cornerRadius: showsTitle ? 0 : 5))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("\(print.title), on \(print.machine)"))
    }
}

private struct EmptyPrints: View {
    let hasMachines: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // a11y: decorative -- the words below say it.
            Image(systemName: "photo.on.rectangle.angled")
                .font(.title2)
                .foregroundStyle(.tint)
                .accessibilityHidden(true)
            Spacer(minLength: 0)
            Text("No prints yet").font(.headline)
            Text(hasMachines ? "What you make appears here." : "Open Mold Studio to add a machine.")
                .font(.caption)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    }
}
