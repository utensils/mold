import MoldClient
import SwiftUI

/// A closer look at one catalog row before installing it -- description,
/// tags, trained words (copyable, the way a component's path is in
/// `ComponentsSheet`), what installing it also fetches, and the row's own
/// licence metadata, which is display only and never an accept gate
/// (design decision 18, M5).
struct CatalogDetailSheet: View {
    let entry: CatalogEntry
    let host: MoldHost
    @Environment(DownloadStore.self) private var downloads
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if let description = entry.description, !description.isEmpty {
                        Text(description).font(.callout)
                    }
                    facts
                    if !entry.tags.isEmpty { labelled("Tags", entry.tags.joined(separator: ", ")) }
                    if !entry.trainedWords.isEmpty {
                        labelled("Trained words", entry.trainedWords.joined(separator: ", "))
                    }
                    if !entry.companionDetails.isEmpty { companions }
                    licence
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            footer
        }
        .padding(20)
        .frame(width: 460, height: 480)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(entry.name).font(.title3.weight(.semibold))
            if let author = entry.author {
                Text("by \(author)").font(.caption).foregroundStyle(.secondary)
            }
            Text("\(entry.family) · \(entry.kind) · \(entry.modality)")
                .font(.caption).foregroundStyle(.secondary)
            // `false` draws nothing -- never an affirmative "Safe" claim.
            if entry.nsfw {
                Text("NSFW").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            }
        }
    }

    private var facts: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let bytes = entry.sizeBytes {
                Text("Size: \(Int64(bytes).formatted(.byteCount(style: .file)))")
            }
            Text("Downloads: \(entry.downloadCount.formatted())")
            if let rating = entry.rating {
                Text("Rating: \(rating.formatted(.number.precision(.fractionLength(1))))")
            }
            Text("Likes: \(entry.likes.formatted())")
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private func labelled(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption.weight(.medium))
            Text(value).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
        }
    }

    /// "Installing this also fetches CLIP-L (246 MB) and SD-VAE-FT-MSE
    /// (335 MB)" -- the honest answer to why a 2 GB checkpoint is a 2.6 GB
    /// download (design S6).
    private var companions: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Also fetches").font(.caption.weight(.medium))
            ForEach(entry.companionDetails, id: \.name) { companion in
                Text(Self.companionLine(companion)).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var licence: some View {
        if Self.showsLicence(entry) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Licence").font(.caption.weight(.medium))
                if let license = entry.license {
                    Text(license).font(.caption).foregroundStyle(.secondary)
                }
                ForEach(Self.licenceRows(entry.licenseFlags), id: \.label) { row in
                    Text("\(row.label): \(row.value)").font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var footer: some View {
        HStack {
            if let pageURL = entry.pageUrl.flatMap(URL.init(string:)) {
                Link("Open Page ↗", destination: pageURL)
            }
            Spacer()
            if case .install = DiscoverRow.resolve(entry) {
                Button("Install") {
                    Task { await downloads.install(entry.id, on: host) }
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
            Button("Done") { dismiss() }
        }
    }

    static func companionLine(_ companion: CatalogCompanionDetail) -> String {
        guard let bytes = companion.sizeBytes else { return companion.name }
        return "\(companion.name) (\(Int64(bytes).formatted(.byteCount(style: .file))))"
    }

    /// The section shows only when there is something to say -- a licence
    /// string, or at least one non-null flag (decision 18, M5).
    static func showsLicence(_ entry: CatalogEntry) -> Bool {
        entry.license != nil || !entry.licenseFlags.isEmpty
    }

    struct LicenceRow: Equatable { let label: String; let value: String }

    /// Empty when every flag is nil -- all-null is the ordinary case and
    /// means no information, never a fabricated "no" (design test 5, M5).
    static func licenceRows(_ flags: CatalogLicenseFlags) -> [LicenceRow] {
        guard !flags.isEmpty else { return [] }
        return [
            LicenceRow(label: "Commercial use", value: triState(flags.commercial)),
            LicenceRow(label: "Derivatives", value: triState(flags.derivatives)),
            LicenceRow(label: "Different licence", value: triState(flags.differentLicense)),
        ]
    }

    static func triState(_ value: Bool?) -> String {
        switch value {
        case true: "Yes"
        case false: "No"
        case nil: "Unknown"
        }
    }
}
