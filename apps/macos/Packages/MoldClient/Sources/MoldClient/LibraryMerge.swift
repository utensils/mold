import Foundation

/// One print, however many machines hold a copy of it.
///
/// Saving a print to This Mac -- one at a time, or Sync All -- leaves the
/// original on the machine that made it, and the merged Library used to show
/// both: every synced print twice, once per machine. The desktop app has
/// always collapsed these (`desktop/src/stores/gallery.ts` `mergeBuckets`,
/// `studio/lib/galleryPrintIdentity.ts`); this is the same rule, so the two
/// apps agree on what "one print" means:
///
/// 1. a copy this app made itself (`links`, from the sync records) joins its
///    source exactly, even under a collision-renamed filename;
/// 2. otherwise the same FILENAME is the same print -- a saved copy keeps the
///    name it had on the machine that rendered it;
/// 3. otherwise seed + exact byte size + model within an hour, which catches
///    older copies whose names diverged, while a genuine re-render reusing a
///    seed much later stays a separate print. Rows without a seed or a size
///    never match this way.
///
/// The LOCAL copy leads when there is one -- it opens without a network and
/// survives the other machine going away -- and every other copy rides along
/// in `copies`.
public enum LibraryMerge {
    /// Mirrors `GALLERY_IDENTITY_WINDOW_SECS`.
    public static let identityWindowSeconds: UInt64 = 3600

    /// `entries` in machine-list order; `links` maps a copy to the print it
    /// was copied from, in either direction.
    public static func merge(
        _ entries: [LibraryEntry], localHost: MoldHost.ID?, links: [PrintID: PrintID] = [:]
    ) -> [LibraryEntry] {
        var groups: [[LibraryEntry]] = []
        var byID: [PrintID: Int] = [:]
        var byFilename: [String: Int] = [:]
        var byIdentity: [String: Int] = [:]

        for entry in entries {
            let identity = Self.identity(of: entry.print)
            var index = links[entry.id].flatMap { byID[$0] }
            if index == nil { index = byFilename[entry.print.filename] }
            if index == nil, let identity, let candidate = byIdentity[identity],
               withinWindow(groups[candidate][0].print, entry.print) {
                index = candidate
            }
            let group: Int
            if let index {
                group = index
                groups[group].append(entry)
            } else {
                group = groups.count
                groups.append([entry])
            }
            byID[entry.id] = group
            // Every name a copy goes by points at the print, so a later copy
            // under either name still joins it.
            byFilename[entry.print.filename] = group
            if let identity, byIdentity[identity] == nil { byIdentity[identity] = group }
        }

        return groups.map { copies in
            let leadIndex = copies.firstIndex { $0.hostID == localHost } ?? 0
            var lead = copies[leadIndex]
            var others = copies
            others.remove(at: leadIndex)
            lead.copies = others
            return lead
        }
    }

    /// `seed:size:model`, or `nil` when the row cannot be matched this way.
    public static func identity(of print: GalleryPrint) -> String? {
        guard let size = print.sizeBytes, size > 0, let seed = seed(of: print) else { return nil }
        return "\(seed):\(size):\(modelSlug(print.metadata.model))"
    }

    /// A synthesized recipe recorded seed 0 as "unknown"; the auto-save
    /// filename still carries the real seed, so it is read from there.
    static func seed(of print: GalleryPrint) -> UInt64? {
        guard print.metadataSynthetic == true else { return print.metadata.seed }
        let pattern = #"-(\d+)-(\d+)(?:-(?:original|upscaled))?\.[a-z0-9]+$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
              let match = regex.firstMatch(
                in: print.filename, range: NSRange(print.filename.startIndex..., in: print.filename)),
              let range = Range(match.range(at: 1), in: print.filename)
        else { return nil }
        return UInt64(print.filename[range])
    }

    static func modelSlug(_ model: String?) -> String {
        let lowered = (model ?? "").lowercased()
        var slug = ""
        var pendingDash = false
        for scalar in lowered.unicodeScalars {
            if ("a"..."z").contains(scalar) || ("0"..."9").contains(scalar) {
                if pendingDash, !slug.isEmpty { slug.append("-") }
                pendingDash = false
                slug.unicodeScalars.append(scalar)
            } else {
                pendingDash = true
            }
        }
        return slug
    }

    static func withinWindow(_ lhs: GalleryPrint, _ rhs: GalleryPrint) -> Bool {
        let gap = lhs.timestamp > rhs.timestamp
            ? lhs.timestamp - rhs.timestamp : rhs.timestamp - lhs.timestamp
        return gap <= identityWindowSeconds
    }
}
