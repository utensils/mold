import MoldClient
import SwiftUI

/// Info (DESIGN.md §5.2): the print's title (editable, blank clears), its
/// star, and everything it recorded -- grouped exactly as the Mac inspector
/// groups it (`PrintDetails`), a heading only where there is something under
/// it, every row copyable. Opens at a third of the screen with the picture
/// still visible above, or straight to full height at accessibility sizes.
struct PrintInfoSheet: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    @Environment(\.dynamicTypeSize) private var size
    let entry: LibraryEntry
    let trashed: Bool
    @State private var title = ""
    @State private var detent: PresentationDetent = .fraction(0.35)

    var body: some View {
        NavigationStack {
            List {
                Section {
                    if canOrganize {
                        HStack(spacing: 8) {
                            TextField("Title", text: $title, prompt: Text(entry.print.displayName))
                                .font(.title3.weight(.semibold))
                                .submitLabel(.done)
                                .onSubmit(saveTitle)
                            Button { library.apply(.favorite(!entry.print.isFavorite), to: [entry]) } label: {
                                Label(entry.print.isFavorite ? "Unfavourite" : "Favourite",
                                      systemImage: entry.print.isFavorite ? "star.fill" : "star")
                                    .labelStyle(.iconOnly)
                            }
                            .buttonStyle(.borderless)
                            .frame(minWidth: 44, minHeight: 44)
                        }
                    } else {
                        Text(entry.print.displayName).font(.title3.weight(.semibold))
                    }
                    if !entry.print.tagList.isEmpty {
                        Text(entry.print.tagList.map { "#\($0)" }.joined(separator: " "))
                            .foregroundStyle(.secondaryText)
                    }
                }
                ForEach(PrintDetails.groups(for: entry)) { group in
                    Section {
                        ForEach(group.rows, id: \.label) { row in
                            DetailRow(row: row)
                        }
                    } header: {
                        SectionHeader(group.title)
                    }
                }
            }
            .navigationTitle("Info")
            .navigationBarTitleDisplayMode(.inline)
        }
        .presentationDetents([.fraction(0.35), .large], selection: $detent)
        .presentationBackgroundInteraction(.enabled(upThrough: .fraction(0.35)))
        .onAppear {
            title = entry.print.title ?? ""
            if size.isAccessibilitySize { detent = .large }
        }
    }

    private var canOrganize: Bool {
        !trashed && entry.everyCopy.allSatisfy { hosts.capabilities[$0.hostID]?.canOrganize == true }
    }

    private func saveTitle() {
        let new = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard new != (entry.print.title ?? "") else { return }
        library.apply(.title(from: entry.print.title ?? "", to: new), to: [entry])
    }
}

/// One recorded fact: plain words, then the value; prose (a prompt) wraps
/// under its label, every row can be copied.
private struct DetailRow: View {
    let row: PrintDetailRow

    var body: some View {
        Group {
            if row.isProse {
                VStack(alignment: .leading, spacing: 4) {
                    Text(row.label).foregroundStyle(.secondaryText)
                    Text(row.value).textSelection(.enabled)
                }
            } else {
                AdaptiveRow { Text(row.label) } value: {
                    Text(row.value).multilineTextAlignment(.trailing)
                }
            }
        }
        .contextMenu {
            Button(row.copyTitle) { UIPasteboard.general.string = row.value }
        }
    }
}
