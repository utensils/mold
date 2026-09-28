import MoldClient
import SwiftUI

/// Said at the top of the Library when some of what it shows is this
/// device's saved copy, because those machines are not answering.
struct OfflineNote: View {
    @Environment(LibraryStore.self) private var library

    var body: some View {
        let names = library.offlineHosts.map(\.name)
        if !names.isEmpty {
            Label {
                Text(names.count == 1
                     ? String(localized: "\(names[0]) isn't answering. Showing its saved prints.")
                     : String(localized: "\(ListFormatter.localizedString(byJoining: names)) aren't answering. Showing their saved prints."))
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                // a11y: decorative -- the sentence says it.
                Image(systemName: "icloud.slash").accessibilityHidden(true)
            }
            .font(.footnote)
            .padding(.horizontal, 12).padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial, in: .rect(cornerRadius: 10))
            .padding(.horizontal, 12)
            .padding(.top, 4)
        }
    }
}
