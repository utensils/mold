import Foundation

// Which request field a retained role fills, and which members are worth
// asking for. Port of `REQUEST_FIELD_FOR_ROLE` and
// `retainedSourceMediaMembersForRequest` (`gallerySourceMedia.ts:206-248`).
public extension RetainedSourceMedia {

    /// The field a role hydrates. Several roles share one -- a video the host
    /// held as a PATH and one it held as bytes both come back as bytes.
    enum Field: Hashable, Sendable {
        case sourceImage, identityImage, identityImages, editImages
        case maskImage, controlImage, audioFile, sourceVideo, extendVideo
        case keyframes
    }

    /// The table, exactly as studio holds it -- MINUS `references`.
    ///
    /// H3's ordered references are not modelled on this app's
    /// `GenerateRequest` at all, so there is no descriptor list for retained
    /// bytes to be matched against. Studio hydrates them only when EVERY
    /// descriptor is descriptor-only; with no descriptors there is nothing
    /// that rule could be true of, so the role is never selected and never
    /// relayed. It is listed here so the omission is a decision rather than
    /// an oversight.
    static let fieldForRole: [String: Field] = [
        "source_image": .sourceImage,
        "identity_image": .identityImage,
        "identity_images": .identityImages,
        "edit_images": .editImages,
        "mask_image": .maskImage,
        "control_image": .controlImage,
        "audio_file": .audioFile,
        "audio_file_path": .audioFile,
        "source_video": .sourceVideo,
        "source_video_path": .sourceVideo,
        "extend_video": .extendVideo,
        "extend_video_path": .extendVideo,
        "keyframes": .keyframes,
    ]

    /// Roles the host will hand over for DOWNLOAD but refuses to hydrate: a
    /// matted picture reused as input would have matting applied twice
    /// (`reusable_role`, `gallery_source_media.rs:118-123`). Asking for one
    /// in a session is a 422 for the whole reuse, so they are dropped here.
    static let notReusableRoles: Set<String> = [
        "matting_processed_source_image", "matting_processed_references",
    ]

    /// The members worth asking for: those whose outgoing field is still
    /// EMPTY. This is what preserves a reattachment someone made by hand --
    /// their picture wins, and the print's retained one is not asked for at
    /// all, so the host's own "target already carries authority for a
    /// selected role" refusal (`RETAINED_MEDIA_REUSE_TARGET_CONFLICT`) can
    /// never fire for something this client chose to send.
    static func members(
        _ members: [Member], forHydrating request: GenerateRequest
    ) -> [Member] {
        members.filter { member in
            guard !notReusableRoles.contains(member.role),
                  let field = fieldForRole[member.role]
            else { return false }
            return request.isVacant(field)
        }
    }
}

public extension GenerateRequest {
    /// Whether this request still lacks byte authority for a retained role.
    func isVacant(_ field: RetainedSourceMedia.Field) -> Bool {
        switch field {
        case .sourceImage: sourceImage == nil
        case .identityImage: idImage == nil
        case .identityImages: idImages?.isEmpty ?? true
        case .editImages: editImages?.isEmpty ?? true
        case .maskImage: maskImage == nil
        case .controlImage: controlImage == nil
        // Both shapes count: the host reads the path form as authority for
        // the same conditioning the bytes would be (`gallery_source_media.rs`
        // `ensure_hydration_target_is_empty`, :620-634).
        case .audioFile: audioFile == nil
        case .sourceVideo: sourceVideo == nil
        case .extendVideo: extendVideo == nil
        case .keyframes: keyframes?.isEmpty ?? true
        }
    }
}
