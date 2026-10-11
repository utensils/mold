import Foundation

private enum ObjectBrokerFailure: Error { case disabled }

extension SecureWebSocketConnection {
    func shouldStagePacket(bulk: Bool) -> Bool {
        objectPolicy != nil && !objectAllocationDisabled && (bulk || !stagedFrames.isEmpty)
    }

    /// True means the complete range was already consumed and only an ACK is due.
    func objectDownloadFailed(_ error: any Error, end: UInt64) async -> Bool {
        guard !Task.isCancelled else { return false }
        if end <= window.next { return true }
        await fail(error)
        return false
    }
    func scheduleObjectFlush() {
        if objectSends.count < 4, stagedFrames.count == 16 {
            flushObjects()
        }
        guard !stagedFrames.isEmpty, objectFlushTask == nil else { return }
        objectFlushTask = Task {
            do { try await Task.sleep(for: .milliseconds(8)); objectFlushTask = nil; flushObjects() }
            catch { }
        }
    }

    /// Enqueue completes before the upload: otherwise sequential body writes
    /// could never fill a batch. All staged frames already consume wire credit.
    func flushObjects() {
        guard failure == nil, !stagedFrames.isEmpty else { return }
        guard objectSends.count < 4 else { scheduleObjectFlush(); return }
        objectFlushTask?.cancel(); objectFlushTask = nil
        let frames = stagedFrames; stagedFrames.removeAll()
        guard let first = frames.first else { return }
        objectSends[first.sequence] = Task {
            defer {
                objectSends.removeValue(forKey: first.sequence); flushObjects()
                let waiter = writer; writer = nil; waiter?.resume()
            }
            do {
                let batch = try SecureObjectBatch(sequence: first.sequence, chunks: frames.map(\.bytes))
                if batch.bytes.count < 131_072 || objectAllocationDisabled {
                    try await sendInlineFrames(frames)
                    return
                }
                let grant: SecureObjectGrant
                do { grant = try await requestObject(batch) }
                catch ObjectBrokerFailure.disabled {
                    objectAllocationDisabled = true
                    try await sendInlineFrames(frames)
                    return
                }
                var request = URLRequest(url: grant.putURL, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 20)
                request.httpMethod = "PUT"; request.httpBody = batch.bytes
                request.httpShouldHandleCookies = false
                for (key, value) in grant.headers { request.setValue(value, forHTTPHeaderField: key) }
                let (responseBytes, response) = try await objectSession.bytes(for: request)
                responseBytes.task.cancel()
                guard let response = response as? HTTPURLResponse,
                      (200..<300).contains(response.statusCode) || response.statusCode == 412 else { throw SecureConnectionError.protocolViolation }
                if let failure { throw failure }
                let envelope = try JSONSerialization.data(withJSONObject: ["a": "data", "v": 3, "sid": sid, "rid": rid,
                    "seq": batch.sequence, "o": grant.reference.json])
                for offset in frames.indices {
                    let seq = batch.sequence + UInt64(offset)
                    pending[seq]?.bytes = envelope; pending[seq]?.sent = .now; pending[seq]?.expiresAt = grant.expiresAt
                }
                try await socket.send(.string(String(decoding: envelope, as: UTF8.self)))
                objectSentBytes += frames.reduce(0) { $0 + $1.bytes.count }
            } catch { if !Task.isCancelled { await fail(error) } }
        }
    }

    private func sendInlineFrames(_ frames: [(sequence: UInt64, bytes: Data)]) async throws {
        for frame in frames {
            if let failure { throw failure }
            guard let pending = pending[frame.sequence] else { continue }
            self.pending[frame.sequence]?.sent = .now
            try await socket.send(.string(String(decoding: pending.bytes, as: UTF8.self)))
            inlineSentBytes += frame.bytes.count
        }
    }

    func requestObject(_ batch: SecureObjectBatch) async throws -> SecureObjectGrant {
        if objectAllocationDisabled { throw ObjectBrokerFailure.disabled }
        let requestID = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
        return try await withCheckedThrowingContinuation { continuation in
            let timeout = Task {
                do {
                    while let pending = objectRequests[requestID] {
                        if ContinuousClock.now >= pending.deadline {
                            objectRequests.removeValue(forKey: requestID)?.continuation.resume(throwing: URLError(.timedOut))
                            return
                        }
                        if ContinuousClock.now >= pending.nextAttempt {
                            objectRequests[requestID]?.nextAttempt = .now + .seconds(5)
                            try await sendJSON(["a": "object", "v": 3, "op": "create", "requestId": requestID, "sid": sid, "rid": rid,
                                "seq": batch.sequence, "count": batch.chunks.count, "bytes": batch.bytes.count, "sha256": batch.checksum])
                        }
                        try await Task.sleep(for: .milliseconds(100))
                    }
                } catch { if !Task.isCancelled { await fail(error) } }
            }
            objectRequests[requestID] = ObjectRequest(batch: batch, continuation: continuation, timeout: timeout,
                deadline: .now + .seconds(30), nextAttempt: .now)
        }
    }

    func receiveObjectReply(_ value: [String: Any]) throws {
        guard let requestID = value["requestId"] as? String else { throw SecureConnectionError.protocolViolation }
        if completedObjectRequests.contains(requestID) { return }
        guard let request = objectRequests[requestID],
              let operation = value["op"] as? String else { throw SecureConnectionError.protocolViolation }
        if operation == "error" {
            guard let code = value["code"] as? String,
                  ["disabled", "capacity", "not_ready", "expired", "conflict", "unavailable"].contains(code)
            else { throw SecureConnectionError.protocolViolation }
            if ["capacity", "not_ready", "unavailable"].contains(code) {
                objectRequests[requestID]?.nextAttempt = .now + .milliseconds(250)
                return
            }
            objectRequests.removeValue(forKey: requestID); request.timeout.cancel()
            rememberCompletedObjectRequest(requestID)
            request.continuation.resume(throwing: code == "disabled" ? ObjectBrokerFailure.disabled : SecureConnectionError.closed)
            return
        }
        guard operation == "created", let policy = objectPolicy
        else { throw SecureConnectionError.protocolViolation }
        let grant = try SecureObjectGrant(value, batch: request.batch, policy: policy, namespace: host.namespace, sid: sid, rid: rid)
        objectRequests.removeValue(forKey: requestID); request.timeout.cancel()
        rememberCompletedObjectRequest(requestID)
        request.continuation.resume(returning: grant)
    }

    private func rememberCompletedObjectRequest(_ requestID: String) {
        completedObjectRequests.append(requestID)
        if completedObjectRequests.count > 32 { completedObjectRequests.removeFirst() }
    }

    /// The socket reader never waits for S3: broker replies and ACKs must keep
    /// flowing while a bounded GET is pending. Noise receives no partial object.
    func receiveObject(sequence: UInt64, value: [String: Any]) throws -> Bool {
        guard let policy = objectPolicy else { throw SecureConnectionError.protocolViolation }
        let object = try SecureObjectReference(value, policy: policy, namespace: host.namespace, sid: sid, rid: rid, direction: "host")
        let end = sequence + UInt64(object.count)
        if end <= window.next { return true }
        guard sequence < window.next + 16, end <= window.next + 16 else { throw SecureConnectionError.protocolViolation }
        if let existing = objectFetches[sequence] {
            guard existing.reference == object else { throw SecureConnectionError.protocolViolation }
            return false
        }
        guard objectFetches.count < 4, objectFetches.values.reduce(object.bytes, { $0 + $1.reference.bytes }) <= 1_048_576
        else { throw SecureConnectionError.capacity }
        let task = Task {
            defer { objectFetches.removeValue(forKey: sequence) }
            do {
                var request = URLRequest(url: object.url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 20)
                request.httpShouldHandleCookies = false
                let (body, response) = try await objectSession.bytes(for: request)
                defer { body.task.cancel() }
                if end <= window.next { try await acknowledge(force: true); return }
                guard let response = response as? HTTPURLResponse, response.statusCode == 200,
                      response.expectedContentLength < 0 || response.expectedContentLength == object.bytes else { throw SecureConnectionError.protocolViolation }
                var bytes = Data(); bytes.reserveCapacity(object.bytes)
                for try await byte in body {
                    guard bytes.count < object.bytes else { throw SecureConnectionError.protocolViolation }
                    bytes.append(byte)
                }
                guard bytes.count == object.bytes else { throw SecureConnectionError.protocolViolation }
                let chunks = try SecureCipherObject.unpack(bytes, count: object.count, checksum: object.checksum)
                if let failure { throw failure }
                for (offset, chunk) in chunks.enumerated() {
                    if try !window.insert(sequence: sequence + UInt64(offset), data: chunk) { objectReceivedBytes += chunk.count }
                }
                packetBytes.append(window.drain(capacity: max(0, 131_078 - packetBytes.count)))
                if let reader = packetReader, let packet = try takePacket() { packetReader = nil; reader.resume(returning: packet) }
                if window.hasGap { if gapSince == nil { gapSince = .now } } else { gapSince = nil }
                try await acknowledge()
            } catch {
                if await objectDownloadFailed(error, end: end) {
                    do { try await acknowledge(force: true) } catch { await fail(error) }
                }
            }
        }
        objectFetches[sequence] = ObjectFetch(reference: object, task: task)
        return false
    }
}
