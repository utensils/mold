import Foundation

// What the metadata MEANS, as opposed to what it stores. Two of the stored
// names are spelled for Foundation rather than for a reader (see
// `editImageSha256S`), so nothing outside this file should reach for them.
public extension OutputMetadata {

    /// Digests of the reference images this print was made from, in request
    /// order.
    var editImageDigests: [String] { editImageSha256S ?? [] }

    /// Digests of every identity photograph, in request order. The plural
    /// form is recorded only for a multi-photograph print, so the singular
    /// stands in for the one-photograph case -- a print that carried a face
    /// either way.
    var identityDigests: [String] {
        if let plural = idImageSha256S, !plural.isEmpty { return plural }
        return idImageSha256.map { [$0] } ?? []
    }

    /// Whether the print was conditioned on a face. The knobs are recorded
    /// ONLY when one actually rode along (`types.rs:3241-3248`), so any of
    /// the four is evidence -- a print made before `id_weight` was recorded
    /// still names its photograph.
    var carriedAFace: Bool {
        idWeight != nil || idStartStep != nil || idImageName != nil
            || !identityDigests.isEmpty
    }

    /// Whether the print was stitched from several clips, however it was
    /// authored. `chain_job_id` is present only for a DURABLE sequence, so it
    /// cannot answer this for an auto-chained one-shot.
    var isChained: Bool { chain != nil || chainJobId != nil }

    /// What a person typed. A sequence's recorded `prompt` is every stage
    /// newline-joined, so reuse takes the FIRST stage rather than restoring a
    /// wall of text that was never one prompt -- mold makes the same
    /// reduction on every other surface, and there is no door back to the
    /// sequence on any of them.
    ///
    /// The stage list is preferred when there is one, because it is the
    /// authored text rather than a substring of the join; splitting the
    /// joined prompt is the fallback for a print whose `chain` block predates
    /// per-stage prompts or is absent.
    var firstStagePrompt: String {
        let joined = prompt ?? ""
        if let first = chain?.stages?.first?.prompt, !first.isEmpty { return first }
        guard outputMode == "sequence" || isChained else { return joined }
        return joined.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? joined
    }
}
