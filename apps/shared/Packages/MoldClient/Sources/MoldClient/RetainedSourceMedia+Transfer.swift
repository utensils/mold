import Foundation
import CryptoKit

public extension RetainedSourceMedia {
    enum TransferSink: String, Codable, Sendable { case memory, privateStaging = "private_staging" }

    struct TransferMember: Codable, Hashable, Sendable {
        public let memberId: String?
        public let role: String
        public let position: String
        public let sink: TransferSink
        public let sizeBytes: Int
        public let sha256: String

        public init(memberId: String? = nil, role: String, position: String,
                    sink: TransferSink = .memory, sizeBytes: Int, sha256: String) {
            self.memberId = memberId
            self.role = role
            self.position = position
            self.sink = sink
            self.sizeBytes = sizeBytes
            self.sha256 = sha256
        }

        public var contentIdentity: Self {
            Self(role: role, position: position, sink: sink, sizeBytes: sizeBytes, sha256: sha256)
        }
    }

    struct TransferOffer: Codable, Hashable, Sendable {
        public let archiveIdentitySha256: String
        public let outputSha256: String
        public let outputSizeBytes: Int
        public let metadata: OutputMetadata?
        public let members: [TransferMember]

        public init(archiveIdentitySha256: String, members: [TransferMember],
                    outputSha256: String, outputSizeBytes: Int, metadata: OutputMetadata?) {
            self.archiveIdentitySha256 = archiveIdentitySha256
            self.outputSha256 = outputSha256
            self.outputSizeBytes = outputSizeBytes
            self.metadata = metadata
            self.members = members
        }
    }

    struct Transfer: Sendable {
        public let archiveIdentitySha256: String
        public let members: [TransferMember]
        public let files: [URL]

        public init(archiveIdentitySha256: String, members: [TransferMember], files: [URL]) {
            self.archiveIdentitySha256 = archiveIdentitySha256
            self.members = members
            self.files = files
        }
    }

    /// Check destination readiness before importing output bytes. An absent
    /// additive block cannot safely promise a source-bearing copy.
    static func preflightMirror(for filename: String, metadata: OutputMetadata?,
                                from origin: any MoldBackend, to target: any MoldBackend) async throws -> String? {
        let offer: TransferOffer
        do { offer = try await origin.retainedMediaTransferOffer(for: filename) }
        catch let error as MoldClientError {
            guard case let .http(status, _, _) = error, status == 404 || status == 405 else { throw error }
            return nil
        }
        guard validTransferDigest(offer.archiveIdentitySha256) else { throw MoldClientError.malformedResponse }
        if !offer.members.isEmpty {
            let capabilities = try await target.capabilities()
            guard capabilities.retainedMediaTransfer?.protocolVersion == 1 else {
                throw transferIncomplete("Update the destination machine to retain this print’s source media.")
            }
        }
        return offer.archiveIdentitySha256
    }

    /// A mirror is complete only after its private inputs have destination authority.
    static func mirrorSources(for sourceFilename: String, metadata: OutputMetadata?,
                              from origin: any MoldBackend, to target: any MoldBackend,
                              as targetFilename: String, expectedSourceArchiveIdentity: String? = nil) async throws {
        try Task.checkCancellation()
        let offer: TransferOffer
        do { offer = try await origin.retainedMediaTransferOffer(for: sourceFilename) }
        catch let error as MoldClientError {
            guard case let .http(status, _, _) = error, status == 404 || status == 405 else { throw error }
            let inventory = try await origin.retainedSourceMedia(for: sourceFilename)
            guard inventory.availability == .unavailableLegacy, !disclosable(metadata) else {
                throw transferIncomplete("Update the source machine to copy this print’s retained media.")
            }
            return
        }
        guard validTransferDigest(offer.archiveIdentitySha256) else { throw MoldClientError.malformedResponse }
        if let expectedSourceArchiveIdentity, expectedSourceArchiveIdentity != offer.archiveIdentitySha256 {
            throw transferIncomplete("The original print changed while copying. Try again.")
        }
        guard !offer.members.isEmpty else {
            if disclosable(metadata) {
                throw transferIncomplete("The source machine no longer has this print’s original media.")
            }
            return
        }
        try validateTransferMembers(offer.members)
        guard validTransferDigest(offer.outputSha256), offer.outputSizeBytes > 0,
              let sourceMetadata = offer.metadata, metadata == nil || sourceMetadata == metadata else {
            throw transferIncomplete("The original print changed while copying. Try again.")
        }
        let destination: TransferOffer
        do { destination = try await target.retainedMediaTransferOffer(for: targetFilename) }
        catch let error as MoldClientError {
            guard case let .http(status, _, _) = error, status == 404 || status == 405 else { throw error }
            throw transferIncomplete("Update the destination machine to retain this print’s source media.")
        }
        guard validTransferDigest(destination.archiveIdentitySha256) else { throw MoldClientError.malformedResponse }
        guard destination.outputSha256 == offer.outputSha256, destination.outputSizeBytes == offer.outputSizeBytes,
              destination.metadata == sourceMetadata else {
            throw transferIncomplete("The copied print no longer matches its original. Try again.")
        }
        if destination.members.map(\.contentIdentity) == offer.members.map(\.contentIdentity) { return }
        guard destination.members.isEmpty else {
            throw transferIncomplete("The destination print has different retained source media.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mold-retained-copy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false,
                                               attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: root) }
        var files: [URL] = []
        for (index, member) in offer.members.enumerated() {
            try Task.checkCancellation()
            guard let id = member.memberId, !id.isEmpty else { throw MoldClientError.malformedResponse }
            let bytes = try await origin.retainedSourceMediaBytes(for: sourceFilename, member: id)
            let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            guard bytes.count == member.sizeBytes, digest == member.sha256 else {
                throw transferIncomplete("The source media changed while copying. Try again.")
            }
            let path = root.appendingPathComponent("\(index).media")
            try bytes.write(to: path, options: .withoutOverwriting)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
            files.append(path)
        }
        try Task.checkCancellation()
        try await target.importRetainedMedia(
            Transfer(archiveIdentitySha256: destination.archiveIdentitySha256,
                     members: offer.members.map(\.contentIdentity), files: files), for: targetFilename)
    }

    static func transferIncomplete(_ reason: String) -> MoldClientError {
        .http(status: 409, code: "RETAINED_MEDIA_COPY_INCOMPLETE", message:
            "The print’s source media was not copied. \(reason)")
    }

    static func validateTransferMembers(_ members: [TransferMember]) throws {
        guard !members.isEmpty, members.count <= maxSessionMembers else { throw MoldClientError.malformedResponse }
        var slots = Set<String>(), total = 0
        for member in members {
            let sum = total.addingReportingOverflow(member.sizeBytes)
            guard member.sizeBytes > 0, !sum.overflow, sum.partialValue <= ResponseCeiling.media,
                  member.role.count <= 160, member.position.count <= 160,
                  !member.role.isEmpty, !member.position.isEmpty,
                  slots.insert("\(member.role)\u{0}\(member.position)").inserted,
                  validTransferDigest(member.sha256)
            else { throw MoldClientError.malformedResponse }
            total = sum.partialValue
        }
    }

    static func validTransferDigest(_ value: String) -> Bool {
        value.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
}
