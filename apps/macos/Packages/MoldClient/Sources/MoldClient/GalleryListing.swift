import Foundation
import os

/// A machine's whole gallery index, minus the rows this client refuses to hold.
///
/// `GalleryPrint` refuses a filename that is not a single safe path component
/// (see `SafeFilename`), and a refusal is an error like any other -- which,
/// decoding `[GalleryPrint]` straight, would lose the entire listing over one
/// row. A library of 4,000 prints going blank because one of them is named
/// oddly is a worse answer than dropping the one, so the array is decoded row
/// by row and only the refusals are dropped.
///
/// Anything else a row throws still fails the listing, exactly as before: a
/// missing required field is this app and that server disagreeing about the
/// wire, not a hostile name.
public struct GalleryListing: Decodable, Sendable {
    public let prints: [GalleryPrint]
    /// What was dropped, so a caller can say so. Empty on every honest host.
    public let rejected: [SafeFilename.Rejected]
    /// Rows a NEWER machine wrote in a shape this build cannot decode. One
    /// such row used to refuse the whole listing -- 1,148 prints gone behind
    /// "it answered something this version of Mold can't read" over a single
    /// provenance field (2026-09-17). Each is logged with its coding path,
    /// which is the part a fix needs.
    public let unreadable: Int

    public init(prints: [GalleryPrint], rejected: [SafeFilename.Rejected] = [], unreadable: Int = 0) {
        self.prints = prints
        self.rejected = rejected
        self.unreadable = unreadable
    }

    public init(from decoder: any Decoder) throws {
        let rows = try decoder.singleValueContainer().decode([Row].self)
        prints = rows.compactMap(\.print)
        rejected = rows.compactMap(\.rejected)
        unreadable = rows.filter(\.isUnreadable).count
        // Every row unreadable is the wire itself disagreeing, not a newer
        // row or two: an empty library would be a lie, so that still fails.
        if unreadable > 0, unreadable == rows.count {
            throw DecodingError.dataCorrupted(.init(
                codingPath: [], debugDescription: "no gallery row decodes"))
        }
        for refusal in rejected {
            // The name itself is the interesting part of the report and it is
            // also attacker-controlled, so it is logged privately -- a console
            // anyone can read is not where someone else's path goes.
            Self.log.error("""
            dropped a gallery row whose filename is not a safe path component \
            (\(refusal.reason.rawValue, privacy: .public)): \
            \(refusal.name, privacy: .private)
            """)
        }
    }

    private static let log = Logger(subsystem: "io.utensils.mold.native",
                                    category: "gallery")

    /// One row, which may be a refusal rather than a print.
    ///
    /// Its `init` never throws on a refusal, so the unkeyed container always
    /// advances -- catching around `container.decode` instead would leave the
    /// index where it was and spin.
    private struct Row: Decodable {
        let print: GalleryPrint?
        let rejected: SafeFilename.Rejected?
        var isUnreadable: Bool { print == nil && rejected == nil }

        init(from decoder: any Decoder) throws {
            do {
                print = try GalleryPrint(from: decoder)
                rejected = nil
            } catch let refusal as SafeFilename.Rejected {
                print = nil
                rejected = refusal
            } catch let error as DecodingError {
                // The path is what a fix needs; the value is a stranger's
                // prompt and stays out of the log.
                GalleryListing.log.error(
                    "dropped a gallery row this build cannot decode: \(DecodingFailure.summary(error), privacy: .public)")
                print = nil
                rejected = nil
            }
        }
    }
}
