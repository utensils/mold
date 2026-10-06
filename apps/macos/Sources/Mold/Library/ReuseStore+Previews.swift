import Foundation
import CryptoKit
import ImageIO
import AppKit
import MoldClient

extension ReuseStore {
    func preview(for reference: GenerationReference, in draft: RenderDraft) -> String? {
        guard pending(for: draft) != nil else { return nil }
        return referencePreviews[reference]
    }

    func loadPreviews(in draft: RenderDraft) async {
        guard let authority = pending(for: draft), let restored,
              RetainedReferenceGuard.canHydrate(references: draft.media.generationReferences,
                original: restored.media.generationReferences, members: authority.members),
              let backend = hosts.backend(for: authority.origin) else { return }
        let fence = currentFence
        let instance = hosts.instanceID(of: authority.origin)
        let members = authority.members.filter { $0.role == "references" }
        for (reference, member) in zip(draft.media.generationReferences, members) {
            guard isCurrent(fence), !Task.isCancelled else { return }
            guard ["image", "named_image"].contains(reference.kind), referencePreviews[reference] == nil else { continue }
            do {
                let bytes: Data
                do { bytes = try await backend.retainedSourceMediaThumbnail(for: authority.filename, member: member.memberId) }
                catch let error as MoldClientError {
                    // Older servers have no preview route. Only small stills may use the original route.
                    guard case .http(404, _, _) = error, member.sizeBytes <= 2 * 1024 * 1024 else { throw error }
                    bytes = try await backend.retainedSourceMediaPreviewBytes(for: authority.filename, member: member.memberId)
                    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
                    guard reference.provenance?.sha256?.lowercased() == digest else {
                        throw MoldClientError.malformedResponse
                    }
                }
                let encoded = await Task.detached(priority: .utility) {
                    Self.smallPreview(bytes)
                }.value
                guard isCurrent(fence), !Task.isCancelled, hosts.instanceID(of: authority.origin) == instance else { return }
                guard let encoded else { throw MoldClientError.malformedResponse }
                referencePreviews[reference] = encoded
                previewFailures.remove(reference)
            } catch {
                if isCurrent(fence), !Task.isCancelled { previewFailures.insert(reference) }
            }
        }
    }

    nonisolated private static func smallPreview(_ bytes: Data) -> String? {
        guard bytes.count <= 2 * 1024 * 1024,
              let source = CGImageSourceCreateWithData(bytes as CFData, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: 320,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: false,
              ] as CFDictionary),
              let data = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else { return nil }
        return data.base64EncodedString()
    }
}
