import Foundation

// The CROSS-HOST half: bytes downloaded from the print's origin, placed
// directly into a request bound for another machine. Port of
// `relayRetainedSourceMedia` (`gallerySourceMedia.ts:254-361`).
//
// A reuse session cannot travel -- the host binds it to its own instance id,
// its own credential and its own archive identity -- so reusing a print on a
// machine that did not make it means carrying the bytes. Paths, pin ids and
// store identities never cross: only the file's contents do.
public extension RetainedSourceMedia {

    /// The request with every downloaded member inlined, or a refusal naming
    /// what went wrong. Returns a FRESH request, so a caller's own
    /// route-planning snapshot stays untouched.
    static func relayed(
        _ downloaded: [(member: Member, bytes: Data)], into request: GenerateRequest
    ) throws -> GenerateRequest {
        var relayed = request
        var grouped: [Field: [Data]] = [:]
        for (member, bytes) in downloaded {
            guard let field = fieldForRole[member.role] else {
                throw RelayFailure.unsupportedRole(member.role)
            }
            guard !notReusableRoles.contains(member.role) else {
                throw RelayFailure.unsupportedRole(member.role)
            }
            // Asked before ANY of it is applied, so a conflict on the last
            // member cannot leave the first three already written in.
            guard relayed.isVacant(field) else { throw RelayFailure.alreadyHeld(field) }
            grouped[field, default: []].append(bytes)
        }
        for (field, bytes) in grouped {
            try relayed.inline(field, bytes)
        }
        return relayed
    }

    /// Why a relay could not be made. Both cases are the client's own
    /// mistake, not the machine's, and both name the thing rather than the
    /// step -- there is nothing to retry blindly.
    enum RelayFailure: LocalizedError, Hashable, Sendable {
        case unsupportedRole(String)
        case alreadyHeld(Field)
        case ambiguous(Field)

        public var errorDescription: String? {
            switch self {
            case .unsupportedRole:
                "This print kept something this version of Mold cannot reuse."
            case .alreadyHeld, .ambiguous:
                "Something is already attached where this print's source media goes."
            }
        }
    }
}

extension GenerateRequest {
    /// Writes retained bytes into one field. Base64 everywhere except
    /// keyframes, which the host retains as the keyframe DOCUMENT rather than
    /// as a picture (`hydrate_selected_members`, `gallery_source_media.rs:700`).
    mutating func inline(
        _ field: RetainedSourceMedia.Field, _ bytes: [Data]
    ) throws {
        func one() throws -> String {
            guard bytes.count == 1 else { throw RetainedSourceMedia.RelayFailure.ambiguous(field) }
            return bytes[0].base64EncodedString()
        }
        switch field {
        case .sourceImage: sourceImage = try one()
        case .identityImage: idImage = try one()
        case .maskImage: maskImage = try one()
        case .controlImage: controlImage = try one()
        case .audioFile: audioFile = try one()
        case .sourceVideo: sourceVideo = try one()
        case .extendVideo: extendVideo = try one()
        case .identityImages: idImages = bytes.map { $0.base64EncodedString() }
        case .editImages: editImages = bytes.map { $0.base64EncodedString() }
        case .keyframes:
            keyframes = try bytes.map {
                do { return try MoldJSON.decoder.decode(KeyframeCondition.self, from: $0) }
                catch { throw RetainedSourceMedia.RelayFailure.unsupportedRole("keyframes") }
            }
        }
    }
}
