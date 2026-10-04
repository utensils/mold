import Foundation
import MoldClient

public extension FakeBackend {
    func retainedMediaTransferOffer(for filename: String) async throws -> RetainedSourceMedia.TransferOffer {
        try await respond("retainedMediaTransferOffer(for:)", [filename])
    }
    func importRetainedMedia(_ transfer: RetainedSourceMedia.Transfer, for filename: String) async throws {
        let _: Void = try await respond("importRetainedMedia(_:for:)", [transfer, filename])
    }
}
