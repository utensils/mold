import CryptoKit
import Foundation

public extension LegacySequenceSettingsReuse {
    /// Authenticated member roles identify each clip. Never alias a later
    /// clip's media into the first clip or infer an order from response order.
    func materialized(_ downloaded: [(member: RetainedSourceMedia.Member, bytes: Data)]) throws -> Self {
        guard downloaded.count <= AuthoredChainRequest.retainedMediaMemberLimit else { throw invalidInputs() }
        var result = self, seen = Set<String>(), total = 0
        let records = metadata.chain?.stages ?? []
        for (member, bytes) in downloaded {
            guard bytes.count <= AuthoredChainRequest.inlineMediaLimitBytes - total,
                  seen.insert(member.role).inserted else { throw invalidInputs() }
            total += bytes.count
            let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            if let expected = member.contentSha256, expected.lowercased() != digest { throw invalidInputs() }
            let parts = member.role.split(separator: ":", omittingEmptySubsequences: false)
            guard parts.count >= 2, let stage = Int(parts[1]), String(stage) == parts[1], result.sequence.stages.indices.contains(stage) else { throw invalidInputs() }
            let encoded = bytes.base64EncodedString()
            switch parts[0] {
            case "stage_source":
                guard parts.count == 2 else { throw invalidInputs() }
                result.sequence.stages[stage].input.sourceImage = encoded
            case "stage_reference":
                guard parts.count == 3, let position = Int(parts[2]), String(position) == parts[2],
                      let references = result.sequence.stages[stage].generationReferences,
                      references.indices.contains(position) else { throw invalidInputs() }
                var reference = references[position]
                guard let expected = reference.provenance?.sha256, expected.lowercased() == digest else { throw invalidInputs() }
                reference.media = .init(authority: "inline", data: encoded)
                result.sequence.stages[stage].input.generationReferences?[position] = reference
            case "stage_keyframe":
                guard parts.count == 3, let position = Int(parts[2]), String(position) == parts[2],
                      let frames = records[stage].keyframes, frames.indices.contains(position),
                      let index = frames[position].frame, index >= 0,
                      index < result.sequence.stages[stage].frames,
                      frames[position].sha256?.lowercased() == digest else { throw invalidInputs() }
                result.sequence.stages[stage].input.keyframes = (result.sequence.stages[stage].input.keyframes ?? [])
                    + [.init(frame: index, image: encoded, name: frames[position].name ?? member.displayName)]
            default: throw invalidInputs()
            }
        }
        for index in records.indices {
            guard result.sequence.stages[index].generationReferences?.allSatisfy({ $0.media.authority == "inline" }) ?? true,
                  (result.sequence.stages[index].keyframes?.count ?? 0) == (records[index].keyframes?.count ?? 0) else { throw invalidInputs() }
            result.sequence.stages[index].input.keyframes?.sort { $0.frame < $1.frame }
        }
        if metadata.sourceImageSha256 != nil, result.sequence.stages[0].input.sourceImage == nil { throw invalidInputs() }
        return result
    }

    private func invalidInputs() -> SequenceDraftError {
        .invalid("This video's original clip inputs cannot be restored exactly, or exceed the supported authored-part media limit. Reopen its original request or attach its inputs explicitly.")
    }
}
