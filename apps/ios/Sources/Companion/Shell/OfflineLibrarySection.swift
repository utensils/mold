import MoldClient
import SwiftUI

/// Settings ▸ Library: what this device keeps so the Library works with no
/// connection -- how much it may use, how much it does, saving every
/// thumbnail now, and emptying it.
struct OfflineLibrarySection: View {
    @Environment(ThumbnailLoader.self) private var thumbnails
    @Environment(LibraryStore.self) private var library
    @Binding var autoSave: Bool
    @AppStorage(Preference.offlineLimit) private var limitMB = OfflineLimit.standard.rawValue
    @State private var used: Int64?

    private var limit: OfflineLimit { OfflineLimit(rawValue: limitMB) ?? .standard }

    var body: some View {
        Section {
            Toggle("Save Finished Prints to Photos", isOn: $autoSave)
            Picker("Offline Storage", selection: $limitMB) {
                ForEach(OfflineLimit.allCases) { Text($0.title).tag($0.rawValue) }
            }
            usage
            if let saving = thumbnails.saving {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Saving thumbnails… \(saving.done) of \(saving.total)")
                    ProgressView(value: Double(saving.done), total: Double(max(saving.total, 1)))
                }
                .accessibilityElement(children: .combine)
                Button("Stop Saving") { thumbnails.cancelSaving() }
            } else {
                Button("Save All Thumbnails for Offline") { thumbnails.save(library.pool) }
                    .disabled(library.pool.isEmpty)
            }
            // Not red: nothing is lost (it all comes back from the machines),
            // and red text on a grouped row was under 4.5:1.
            Button("Empty Now") {
                Task {
                    await thumbnails.emptyCaches()
                    await measure()
                }
            }
        } header: {
            SectionHeader(String(localized: "Library"))
        } footer: {
            Text("The Library, its thumbnails and the prints you open are kept on this device, so you can browse them without a connection. Emptying removes nothing from your machines.")
                .foregroundStyle(.secondaryText)
        }
        .task { await measure() }
        .onChange(of: limitMB) {
            Task {
                await thumbnails.apply(limit)
                await measure()
            }
        }
        .onChange(of: thumbnails.saving?.done) { Task { await measure() } }
    }

    private var usage: some View {
        AdaptiveRow {
            Text("Used")
        } value: {
            Text(used.map { String(localized: "\(ByteCountFormatter.string(fromByteCount: $0, countStyle: .file)) of \(limit.title)") }
                 ?? String(localized: "Measuring…"))
                .monospacedDigit()
        }
        .accessibilityElement(children: .combine)
    }

    private func measure() async { used = await thumbnails.diskBytes() }
}
