import Foundation

/// Renaming or removing a tag on the prints that carry it.
///
/// A transformation over rows, so the store's job is to hand it the rows and
/// keep the answer -- and so "renaming to a tag a print already has must not
/// give it the tag twice" is a test rather than a thing to notice in review.
/// Comparison is case-insensitive throughout, because that is what the machine
/// does with a tag.
public enum TagRewrite {
    /// The same rows with `name` rewritten to `replacement`, or removed when
    /// that is `nil`. Rows that do not carry it are returned untouched.
    public static func applied(to entries: [LibraryEntry], name: String,
                               replacement: String?) -> [LibraryEntry] {
        entries.map { entry in
            guard entry.print.tagList.contains(where: {
                $0.caseInsensitiveCompare(name) == .orderedSame
            }) else { return entry }
            var mutable = GalleryPrint.Mutable(entry.print)
            var tags = mutable.tags ?? []
            tags.removeAll { $0.caseInsensitiveCompare(name) == .orderedSame }
            if let replacement,
               !tags.contains(where: { $0.caseInsensitiveCompare(replacement) == .orderedSame }) {
                tags.append(replacement)
            }
            mutable.tags = tags
            return entry.replacingPrint(mutable.build())
        }
    }
}
