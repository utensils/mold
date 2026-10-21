import CryptoKit
import Foundation

extension RetainedSourceMedia {
    /// Only the exact offered output may corroborate facts omitted from an
    /// older archive. Never use a filename or unverified embedded recipe.
    static func mirrorRecipeMatches(_ destination: TransferOffer, source: TransferOffer,
                                    filename: String, origin: any MoldBackend, downloadedOutput: URL? = nil) async throws -> Bool {
        if mirrorMetadataMatches(destination.canonicalMetadataJSON, source.canonicalMetadataJSON) { return true }
        let suffix = URL(fileURLWithPath: filename).pathExtension.lowercased()
        guard ["png", "jpg", "jpeg", "gif"].contains(suffix),
              source.outputSizeBytes > 0, source.outputSizeBytes <= ResponseCeiling.media,
              let metadata = source.metadata, let destinationMetadata = destination.metadata else { return false }
        let missingScheduler = (metadata.scheduler == nil) != (destinationMetadata.scheduler == nil)
        let missingTransparency = (metadata.transparentBackground == nil) != (destinationMetadata.transparentBackground == nil)
        guard missingScheduler || missingTransparency else { return false }
        let file: URL
        if let downloadedOutput { file = downloadedOutput }
        else { file = try await origin.mediaFile(filename, trashed: false) }
        // A supplied file is borrowed; its caller owns cleanup.
        defer { if downloadedOutput == nil { try? FileManager.default.removeItem(at: file) } }
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var digest = SHA256(), count = 0
        while let chunk = try handle.read(upToCount: 1_024 * 1_024), !chunk.isEmpty {
            try Task.checkCancellation()
            let sum = count.addingReportingOverflow(chunk.count)
            guard !sum.overflow, sum.partialValue <= source.outputSizeBytes else { return false }
            count = sum.partialValue
            digest.update(data: chunk)
        }
        let hash = digest.finalize().map { String(format: "%02x", $0) }.joined()
        guard count == source.outputSizeBytes, hash == source.outputSha256,
              let embedded = try EmbeddedPrintMetadata.json(in: file, named: filename,
                                                            metadataCeiling: 2 * 1_024 * 1_024) else { return false }
        return mirrorMetadataMatches(source.canonicalMetadataJSON,
                                     destination.canonicalMetadataJSON,
                                     verifiedEmbeddedRecipe: embedded)
    }
}
