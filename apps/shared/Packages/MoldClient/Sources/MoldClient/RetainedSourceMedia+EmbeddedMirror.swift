import CryptoKit
import Foundation

extension RetainedSourceMedia {
    /// Only the exact offered output may corroborate facts omitted from an
    /// older archive. Never use a filename or unverified embedded recipe.
    static func mirrorRecipeMatches(_ destination: OutputMetadata?, source: TransferOffer,
                                    filename: String, origin: any MoldBackend) async throws -> Bool {
        if mirrorMetadataMatches(destination, source.metadata) { return true }
        let suffix = URL(fileURLWithPath: filename).pathExtension.lowercased()
        guard ["png", "jpg", "jpeg", "gif"].contains(suffix),
              source.outputSizeBytes > 0, source.outputSizeBytes <= ResponseCeiling.media,
              let metadata = source.metadata, let destination else { return false }
        let missingScheduler = (metadata.scheduler == nil) != (destination.scheduler == nil)
        let missingTransparency = (metadata.transparentBackground == nil) != (destination.transparentBackground == nil)
        guard missingScheduler || missingTransparency else { return false }
        let file = try await origin.mediaFile(filename, trashed: false)
        defer { try? FileManager.default.removeItem(at: file) }
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
        return mirrorMetadataMatches(try MoldJSON.encoder.encode(metadata),
                                     try MoldJSON.encoder.encode(destination),
                                     verifiedEmbeddedRecipe: embedded)
    }
}
