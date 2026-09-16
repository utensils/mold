import MoldClient
import MoldStyle
import SwiftUI

/// The day-sectioned grid of prints.
struct LibraryGrid: View {
    let sections: [LibrarySection]
    let hosts: [MoldHost]
    let edge: CGFloat
    let showsHostBadges: Bool
    @Binding var selection: PrintID?

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, alignment: .leading, spacing: 16) {
                ForEach(sections) { section in
                    Section {
                        ForEach(section.items) { item in
                            cell(item)
                        }
                    } header: {
                        header(section)
                    }
                }
            }
            .padding(16)
        }
        .scrollContentBackground(.hidden)
    }

    private var columns: [GridItem] {
        [GridItem(.adaptive(minimum: edge, maximum: .infinity), spacing: 12)]
    }

    @ViewBuilder private func cell(_ item: LibraryEntry) -> some View {
        if let host = hosts.first(where: { $0.id == item.hostID }) {
            LibraryCell(
                item: item, host: host, edge: edge,
                isSelected: selection == item.id,
                showsHostBadge: showsHostBadges
            )
            .onTapGesture { selection = item.id }
        }
    }

    private func header(_ section: LibrarySection) -> some View {
        HStack {
            Text(LibraryGrouping.title(for: section.day))
                .font(.headline)
            Text(section.items.count.formatted())
                .font(.subheadline)
                .monospacedDigit()
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.top, 8)
        // A material keeps the heading readable while tiles scroll under it.
        .background(.bar)
    }
}
