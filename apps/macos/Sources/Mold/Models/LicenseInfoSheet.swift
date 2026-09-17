import MoldClient
import SwiftUI

/// A read-only look at a licence already on this machine's own `GET
/// /api/licenses` listing -- name, the server's summary, which models it
/// gates, whether it is accepted, and a link to the terms.
///
/// Deliberately a separate small view rather than `LicenseSheet`
/// parameterised: that sheet is typed on `LicenseRefusal` (the 403 payload),
/// which carries neither `accepted` nor `requiredBy`/`requiredByStyles` --
/// fields only the full `ThirdPartyLicense` listing has. Bending one type to
/// answer both shapes would cost more than this file does.
struct LicenseInfoSheet: View {
    let license: ThirdPartyLicense
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            ScrollView {
                Text(license.summary)
                    .font(.callout)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 140)
            Text("Required by: \(Self.gatedNames(license))")
                .font(.caption)
                .foregroundStyle(.secondary)
            Link("Read the terms ↗", destination: URL(string: license.canonical) ?? URL(string: "about:blank")!)
            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 420, height: 340)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(license.name).font(.title2.weight(.semibold))
            Label(license.accepted ? "Accepted" : "Not accepted", systemImage: license.accepted ? "checkmark.circle.fill" : "circle")
                .font(.callout)
                .foregroundStyle(license.accepted ? .green : .secondary)
        }
    }

    /// The registry's own words when this server sends them, else the bare
    /// manifest names -- the same fallback `requiredByStyles` documents.
    static func gatedNames(_ license: ThirdPartyLicense) -> String {
        if let styles = license.requiredByStyles, !styles.isEmpty {
            return styles.map(\.name).joined(separator: ", ")
        }
        return license.requiredBy.joined(separator: ", ")
    }
}
