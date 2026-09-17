import MoldClient
import MoldStyle
import SwiftUI

/// The tag chips for a draft: what was typed, the one the title adds, and a
/// field to add more.
///
/// Bound to the DRAFT rather than to a selection -- `TagEditor` and
/// `CollectionsField` both answer "what does every selected print share",
/// which a draft does not have, so they are not reused here; only the chip
/// LOOK is.
struct FileUnderTagsRow: View {
    @Binding var tags: [String]
    let title: String
    @Binding var autoTagTitle: Bool

    @State private var adding = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !tags.isEmpty || ghostTag != nil {
                WrappingHStack(horizontalSpacing: 4, verticalSpacing: 4, alignment: .center) {
                    ForEach(tags, id: \.self) { tag in
                        chip(tag) { tags.removeAll { $0 == tag } }
                    }
                    if let ghostTag {
                        chip(ghostTag, ghost: true) { autoTagTitle = false }
                    }
                }
            }
            TextField("Add a tag", text: $adding)
                .textFieldStyle(.roundedBorder)
                .controlSize(.small)
                .onSubmit(addTag)
        }
    }

    /// The title's slug, shown only while it would actually be added: the
    /// switch is on, the title has a usable slug, and it is not already one
    /// of the typed tags. This is `compose_client_tags`'s `auto_tagged`
    /// disclosure -- a tag the user did not type must be visible here, not a
    /// surprise discovered later in the Library.
    private var ghostTag: String? {
        guard autoTagTitle, let slug = ClientTags.titleSlug(title) else { return nil }
        guard !tags.contains(where: { $0.caseInsensitiveCompare(slug) == .orderedSame }) else {
            return nil
        }
        return slug
    }

    private func addTag() {
        let trimmed = adding.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        tags = ClientTags.normalize(tags + [trimmed])
        adding = ""
    }

    private func chip(_ tag: String, ghost: Bool = false, remove: @escaping () -> Void) -> some View {
        HStack(spacing: 3) {
            Text(tag)
            if ghost {
                Text("from title").font(.caption2).foregroundStyle(.tertiary)
            }
            Button(action: remove) { Image(systemName: "xmark") }
                .buttonStyle(.plain)
                .accessibilityLabel("Remove the tag “\(tag)”")
        }
        .font(.caption)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(Chrome.wellFill, in: Capsule())
        .opacity(ghost ? 0.7 : 1)
        .help(ghost ? "Added from the title -- remove to stop auto-tagging" : "Remove this tag")
    }
}
