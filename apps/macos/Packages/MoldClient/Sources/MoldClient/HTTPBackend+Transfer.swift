import Foundation

/// Sending a HELD job to another machine.
///
/// Three calls the client orchestrates itself: there is no server-to-server
/// push (design M6 fact 4, `routes.rs:7548-7599`, `:2973-2994`).
public extension HTTPBackend {
    /// Exports a HELD row as a portable request with its media inlined.
    ///
    /// The bytes are OPAQUE and stay that way: the destination's admission
    /// body splices them in verbatim, because `GenerateRequest`'s
    /// hand-written `encode(to:)` enumerates a fixed field list, and a
    /// decode-and-re-encode would drop every field this build does not
    /// know -- including, on a request with references, the inlined
    /// reference media the export exists to carry (`queue_transfer.rs:34-97`).
    /// 409 unless the row is held; 422 with the machine's own sentence for a
    /// machine-local LoRA, an EXR directory or a mesh workflow.
    ///
    /// The destination's `MAX_REQUEST_BODY_BYTES` (64 MiB, `lib.rs:178`) is
    /// this call's practical ceiling too, since the admission body carries
    /// the same bytes this answers with: past it the destination answers
    /// 413, nothing is admitted, and the source row stays held.
    func exportHeldJob(_ authority: QueueAuthority) async throws -> Data {
        try await postRaw(transferExportPath(authority.jobId), body: authority)
    }

    /// Admits one exported request HERE, fenced on this machine's identity.
    ///
    /// The body is assembled from bytes rather than encoded, for the reason
    /// above. The `x-mold-destination-instance` header is required, and a
    /// mismatch is `409 TRANSFER_DESTINATION_CHANGED` (`routes.rs:2973-2994`)
    /// -- the fence against a destination that restarted between the picker
    /// and the click.
    func admitTransfer(
        clientBatchId: String, portable: Data, destinationInstance: String
    ) async throws -> BatchStatus {
        let request = try transferAdmissionRequest(
            clientBatchId: clientBatchId, portable: portable, destinationInstance: destinationInstance)
        let data = try await bytes(for: request)
        do {
            return try MoldJSON.decoder.decode(BatchStatus.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
    }

    /// Cancels the held source row, and ONLY after the destination accepted.
    /// 409 if it is no longer held -- someone else resumed or removed it,
    /// and nothing was changed (`routes.rs:7584-7599`).
    func completeTransfer(_ authority: QueueAuthority) async throws {
        _ = try await postRaw(transferCompletePath(authority.jobId), body: authority)
    }
}

/// URL and body construction, split out so a test can pin both without a
/// network call -- the `historyPath` / `deviceMutationPath` precedent.
extension HTTPBackend {
    func transferExportPath(_ jobId: String) -> String {
        "/api/queue/\(escaped(jobId))/transfer"
    }

    func transferCompletePath(_ jobId: String) -> String {
        "/api/queue/\(escaped(jobId))/transfer/complete"
    }

    static let transferAdmitPath = "/api/generation-batches/transfer"

    /// The full `POST /api/generation-batches/transfer` request.
    func transferAdmissionRequest(
        clientBatchId: String, portable: Data, destinationInstance: String
    ) throws -> URLRequest {
        var request = self.request(Self.transferAdmitPath, method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(destinationInstance, forHTTPHeaderField: "x-mold-destination-instance")
        request.httpBody = try Self.transferAdmissionBody(clientBatchId: clientBatchId, portable: portable)
        return request
    }

    /// `{"client_batch_id":<escaped>,"requests":[<the export's response
    /// body, byte for byte>]}`. `portable` is never parsed -- see
    /// `exportHeldJob`. The id is passed through `MoldJSON.encoder` on a
    /// one-field struct so it is JSON-escaped rather than interpolated raw,
    /// which would either break the JSON or let an id inject fields.
    static func transferAdmissionBody(clientBatchId: String, portable: Data) throws -> Data {
        let prefix = "{\"client_batch_id\":\""
        let suffix = "\"}"
        let encoded = try MoldJSON.encoder.encode(ClientBatchIdField(clientBatchId: clientBatchId))
        guard let text = String(data: encoded, encoding: .utf8),
              text.hasPrefix(prefix), text.hasSuffix(suffix)
        else { throw MoldClientError.malformedResponse }
        let escapedId = text.dropFirst(prefix.count).dropLast(suffix.count)
        var body = Data("{\"client_batch_id\":\"\(escapedId)\",\"requests\":[".utf8)
        body.append(portable)
        body.append(Data("]}".utf8))
        return body
    }
}

private struct ClientBatchIdField: Encodable {
    let clientBatchId: String
}
