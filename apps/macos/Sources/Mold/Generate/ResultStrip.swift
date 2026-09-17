import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// The other pictures a batch made, under the one that is showing large.
///
/// Absent for a batch of one -- `RunCanvas` only builds this row when there
/// is more than one result, so the ordinary case renders exactly as it did
/// before a batch could be more than one child.
struct ResultStrip: View {
    let results: [BatchResult]
    let host: MoldHost?
    let actions: ResultActions
    @Binding var selected: Int

    @Environment(HostStore.self) private var hosts
    @State private var thumbnails: [Int: NSImage] = [:]
    /// A caret in the prompt field has the better claim on an unmodified
    /// arrow key. `PromptPanel`'s prompt field now publishes this the same
    /// way the Library's title and tag fields already do, so one gate
    /// covers both panes.
    @FocusedValue(\.editingText) private var editingText: Bool?
    /// And so does a focused `Slider` or `Stepper`, which AppKit alone can
    /// answer for (`ArrowKeyClaim`). Re-read as the key window updates, and
    /// written only when it CHANGES, so the strip's body is not churned.
    @State private var responderClaimsArrows = false

    var body: some View {
        HStack(spacing: 8) {
            ForEach(results.indices, id: \.self) { index in
                thumbnail(index)
            }
        }
        .task { await loadThumbnails() }
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didUpdateNotification)) { _ in
            let claimed = ArrowKeyClaim.isClaimedNow
            if claimed != responderClaimsArrows { responderClaimsArrows = claimed }
        }
    }

    private func thumbnail(_ index: Int) -> some View {
        Button { selected = index } label: {
            ZStack {
                if let image = thumbnails[index] {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } else {
                    Chrome.wellFill
                }
            }
            .frame(width: 56, height: 56)
            .clipShape(RoundedRectangle(cornerRadius: Chrome.thumbnailRadius, style: .continuous))
            .overlay {
                if index == selected {
                    RoundedRectangle(cornerRadius: Chrome.thumbnailRadius, style: .continuous)
                        .strokeBorder(Color.accentColor, lineWidth: 2)
                }
            }
        }
        .buttonStyle(.plain)
        .keyboardShortcut(shortcut(for: index))
        .accessibilityLabel("Result \(index + 1) of \(results.count)")
        .accessibilityAddTraits(index == selected ? .isSelected : [])
        .resultContextMenu(results[index], actions: actions)
    }

    private func shortcut(for index: Int) -> KeyboardShortcut? {
        Self.key(for: index, selected: selected,
                 arrowsAreClaimed: editingText == true || responderClaimsArrows)
            .map { KeyboardShortcut($0, modifiers: []) }
    }

    private func loadThumbnails() async {
        guard let host else { return }
        let backend = hosts.backend(for: host)
        for (index, result) in results.enumerated() {
            guard thumbnails[index] == nil, let filename = result.filename else { continue }
            guard let data = try? await backend.media(filename, trashed: false) else { continue }
            thumbnails[index] = NSImage(data: data)
        }
    }
}

extension ResultStrip {
    /// The left arrow belongs to the thumbnail one step back, the right arrow
    /// to the one step forward -- so an arrow key always does exactly what
    /// clicking that neighbour would, and a thumbnail at either end simply
    /// carries no shortcut rather than wrapping around.
    ///
    /// `arrowsAreClaimed` is the ONE question (`ArrowKeyClaim`): a window
    /// -scoped key equivalent is checked before the first responder sees the
    /// key, so the strip stands down entirely while a caret, a `Slider` or a
    /// `Stepper` is using the same two keys. Pure, so that stand-down is
    /// pinned by a test rather than by a view (finding 02#15).
    static func key(for index: Int, selected: Int, arrowsAreClaimed: Bool) -> KeyEquivalent? {
        guard !arrowsAreClaimed else { return nil }
        if index == selected - 1 { return .leftArrow }
        if index == selected + 1 { return .rightArrow }
        return nil
    }
}
