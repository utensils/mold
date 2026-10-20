import Foundation

public extension GenerateRequest {
    /// Chain admission seals original bytes itself. Ordinary one-use upload
    /// handles cannot be reused across stages or passed into this route.
    func authoredChainInputs() throws -> Self {
        var copy = self
        if let originals = pendingReferenceInputs {
            guard let references = originals.canonicalReferences else {
                throw SequenceDraftError.invalid("Joined clips need canonical inline reference media. Use an image or audio reference; fresh video references require normalization before they can be chained.")
            }
            copy.references = references
            copy.pendingReferenceInputs = nil
        }
        guard copy.references?.allSatisfy({ $0.media.authority == "inline"
            && $0.media.data?.isEmpty == false && $0.media.handle == nil && $0.media.path == nil }) != false else {
            throw SequenceDraftError.invalid("Joined clips require original inline reference media. Reattach the reference before generating.")
        }
        return copy
    }
}
