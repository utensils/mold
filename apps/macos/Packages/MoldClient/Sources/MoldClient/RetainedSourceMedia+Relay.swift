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

    /// What a relay would put in the admission body, from the sizes the
    /// INVENTORY already reported -- so it is answerable before a byte is
    /// downloaded.
    ///
    /// The bytes are base64'd (+1/3) into every sibling and the whole
    /// `BatchAdmission` goes out as one body, so four copies of a render
    /// conditioned on a 400 MB retained clip is ~2.1 GB in a single POST.
    /// Every machine refuses a body past `MAX_REQUEST_BODY_BYTES` anyway
    /// (`lib.rs:178`), so this is the host's own limit asked early rather
    /// than a number this app invented.
    static func relayBodyBytes(_ members: [Member], copies: Int) -> Int {
        let raw = members.reduce(0) { $0 + max($1.sizeBytes, 0) }
        return (raw + 2) / 3 * 4 * max(copies, 1)
    }

    /// The refusal, when a relay would not fit, asked BEFORE downloading.
    static func relayRefusal(_ members: [Member], copies: Int) -> RelayFailure? {
        let bytes = relayBodyBytes(members, copies: copies)
        guard bytes > RequestBodyLimit.bytes else { return nil }
        return .tooLarge(bytes: bytes, copies: max(copies, 1))
    }

    /// Why a relay could not be made. Every case is the client's own
    /// situation, not the machine's, and each names the thing rather than the
    /// step -- there is nothing to retry blindly.
    enum RelayFailure: LocalizedError, Hashable, Sendable {
        case unsupportedRole(String)
        case alreadyHeld(Field)
        case ambiguous(Field)
        case tooLarge(bytes: Int, copies: Int)

        public var errorDescription: String? {
            switch self {
            case .unsupportedRole:
                return "This print kept something this version of Mold cannot reuse."
            case .alreadyHeld, .ambiguous:
                return "Something is already attached where this print's source "
                    + "media goes."
            case let .tooLarge(bytes, copies):
                let size = Int64(bytes).formatted(.byteCount(style: .binary))
                return copies > 1
                    ? "Carrying this print's source media to \(copies) copies would "
                        + "send \(size), and a machine accepts at most "
                        + "\(RequestBodyLimit.sentence). Make one at a time, or "
                        + "attach the picture yourself."
                    : "This print's source media is \(size), and a machine accepts "
                        + "at most \(RequestBodyLimit.sentence). Attach the picture "
                        + "yourself."
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
