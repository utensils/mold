import SwiftUI

/// Toolbars default to icon-only labels. Filter state must remain readable
/// without opening the menu; the system still owns its bezel and disclosure.
struct LibraryFilterLabel: View {
    let title: String
    let systemImage: String

    var body: some View {
        Label {
            Text(title)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: 160)
        } icon: {
            Image(systemName: systemImage)
        }
        .labelStyle(.titleAndIcon)
        .fixedSize(horizontal: true, vertical: false)
    }
}
