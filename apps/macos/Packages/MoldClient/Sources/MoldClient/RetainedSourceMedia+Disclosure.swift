import Foundation

// Whether an UNAVAILABLE answer is worth telling someone about, and what to
// say. Port of `retainedSourceMediaDisclosable` / `retainedSourceMediaDisclosure`
// (`studio/api/gallerySourceMedia.ts:154-188`).
public extension RetainedSourceMedia {

    /// This decides DISCLOSURE, never whether to ask.
    ///
    /// A text-to-image print retains no media set, but its archive entry
    /// still exists with no pins -- which the host can only report as
    /// `unavailable_legacy`. It cannot tell that from a genuinely pre-feature
    /// print, so the client stays quiet unless the print's OWN metadata says
    /// conditioning bytes shipped. Otherwise every picture ever made would be
    /// told to reattach a source it never had.
    ///
    /// The markers mirror the host's downloadable roles (`downloadable_role`,
    /// `gallery_source_media.rs:106-116`). `source_image_name` is
    /// deliberately EXCLUDED: it is a name with no bytes, and MiniMax H3 sets
    /// it alone.
    static func disclosable(_ metadata: OutputMetadata?) -> Bool {
        guard let metadata else { return false }
        return metadata.sourceImageSha256 != nil
            || metadata.idImageSha256 != nil
            || metadata.sourceVideoPath != nil
            || metadata.audioFilePath != nil
            || metadata.extendVideoPath != nil
            || metadata.extendOverlapFrames != nil
            || !metadata.editImageDigests.isEmpty
            || !(metadata.idImageSha256S ?? []).isEmpty
            || !(metadata.references ?? []).isEmpty
            || !(metadata.keyframes ?? []).isEmpty
    }

    /// The sentence, shared with web, desktop and the phone -- verbatim, so a
    /// person who has read it once on another surface reads the same words
    /// here. `available` says nothing, and neither does a state this build
    /// cannot name.
    static func disclosure(_ availability: Availability) -> String? {
        switch availability {
        case .available, .unknown:
            nil
        case .unavailableLegacy:
            "This older print did not retain its original source media. "
                + "Reattach it before developing."
        case .unavailableMissingOrCorrupt:
            "This print\u{2019}s retained source media is missing or damaged. "
                + "Reattach it before developing."
        case .unavailableAuth:
            "Connect this machine with an API key to restore its private source media."
        }
    }
}
