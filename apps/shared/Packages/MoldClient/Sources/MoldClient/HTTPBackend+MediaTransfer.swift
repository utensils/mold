import Foundation

public extension HTTPBackend {
    func retainedMediaTransferOffer(for filename: String) async throws -> RetainedSourceMedia.TransferOffer {
        try await get(retainedSourceMediaPath(filename) + "/transfer")
    }

    func importRetainedMedia(_ transfer: RetainedSourceMedia.Transfer, for filename: String) async throws {
        var sending = request(retainedSourceMediaPath(filename) + "/transfer")
        sending.httpMethod = "PUT"
        sending.setValue("application/vnd.mold.retained-media-transfer", forHTTPHeaderField: "Content-Type")
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("mold-retained-envelope-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try transfer.writeBody(to: temporary)
        let data = try await upload(sending, fromFile: temporary)
        struct Answer: Decodable { let archiveIdentitySha256: String; let memberCount: Int }
        guard let answer = try? MoldJSON.decoder.decode(Answer.self, from: data),
              answer.archiveIdentitySha256 == transfer.archiveIdentitySha256,
              answer.memberCount == transfer.members.count else { throw MoldClientError.malformedResponse }
    }
}

extension RetainedSourceMedia.Transfer {
    public func writeBody(to url: URL) throws {
        try RetainedSourceMedia.validateTransferMembers(members)
        guard members.count == files.count, RetainedSourceMedia.validTransferDigest(archiveIdentitySha256) else { throw MoldClientError.malformedResponse }
        struct Descriptor: Encodable {
            let archiveIdentitySha256: String
            let members: [RetainedSourceMedia.TransferMember]
        }
        let json = try MoldJSON.encoder.encode(Descriptor(archiveIdentitySha256: archiveIdentitySha256, members: members))
        guard json.count <= 128 * 1024 else { throw MoldClientError.malformedResponse }
        guard FileManager.default.createFile(atPath: url.path, contents: nil,
                                              attributes: [.posixPermissions: 0o600]) else { throw CocoaError(.fileWriteUnknown) }
        let output = try FileHandle(forWritingTo: url)
        defer { try? output.close() }
        try withUnsafeBytes(of: UInt32(json.count).bigEndian) { try output.write(contentsOf: $0) }
        try output.write(contentsOf: json)
        for (member, file) in zip(members, files) {
            let input = try FileHandle(forReadingFrom: file)
            defer { try? input.close() }
            var written = 0
            while let chunk = try input.read(upToCount: 1024 * 1024), !chunk.isEmpty {
                try Task.checkCancellation()
                let sum = written.addingReportingOverflow(chunk.count)
                guard !sum.overflow, sum.partialValue <= member.sizeBytes else { throw MoldClientError.malformedResponse }
                try output.write(contentsOf: chunk)
                written = sum.partialValue
            }
            guard written == member.sizeBytes else { throw MoldClientError.malformedResponse }
        }
    }
}
