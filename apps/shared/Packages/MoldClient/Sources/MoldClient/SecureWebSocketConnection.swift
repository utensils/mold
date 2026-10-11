import Foundation

/// Browser-compatible v3 admission and bounded reliable ciphertext framing. There are
/// no credentials on the WebSocket URL or HTTP upgrade, and no plaintext API requests.
actor SecureWebSocketConnection: SecureRecordConnection {
    struct PendingFrame { var bytes: Data; var sent: ContinuousClock.Instant?; var tries: Int; var expiresAt: UInt64? }
    struct PacketSend { let bytes: Data; let bulk: Bool; let continuation: CheckedContinuation<Void, any Error> }
    struct ObjectRequest { let batch: SecureObjectBatch; let continuation: CheckedContinuation<SecureObjectGrant, any Error>; let timeout: Task<Void, Never>; let deadline: ContinuousClock.Instant; var nextAttempt: ContinuousClock.Instant }
    struct ObjectFetch { let reference: SecureObjectReference; let task: Task<Void, Never> }
    let usesTLS: Bool
    let socket: URLSessionWebSocketTask
    let urlSession: URLSession
    let objectSession: URLSession
    let host: SecureHost
    let identity: SecureIdentity
    let invitation: SecureInvitation?
    let trustedObjectOrigin: URL
    var objectPolicy: SecureObjectPolicy?
    var objectAllocationDisabled = false
    var stagedFrames: [(sequence: UInt64, bytes: Data)] = []
    var objectFlushTask: Task<Void, Never>?
    var objectSends: [UInt64: Task<Void, Never>] = [:]
    var objectReservations: [UInt64: UInt64] = [:]
    var objectFetches: [UInt64: ObjectFetch] = [:]
    var objectRequests: [String: ObjectRequest] = [:]
    var completedObjectRequests: [String] = []
    var ackTask: Task<Void, Never>?
    var lastSentAcknowledgement: Data?
    // Session-local aggregate counters contain no request, credential or content.
    var inlineSentBytes = 0, objectSentBytes = 0, inlineReceivedBytes = 0, objectReceivedBytes = 0
    var noise: SecureNoiseSession?
    var sid = "", rid = ""
    var failure: (any Error)?
    var readerTask: Task<Void, Never>?, timerTask: Task<Void, Never>?, senderTask: Task<Void, Never>?
    var sendQueue: [PacketSend] = []
    var sending: PacketSend?
    var pending: [UInt64: PendingFrame] = [:]
    var outgoing: UInt64 = 0, acknowledged: UInt64 = 0, lastAck: UInt64?, ackSequence: UInt64 = 0
    var remoteCredit = SecureRelayWindow.capacity
    var writer: CheckedContinuation<Void, any Error>?
    var window = SecureRelayWindow()
    var packetBytes = Data()
    var packetReader: CheckedContinuation<Data, any Error>?
    var advertisedNext: UInt64?, advertisedCredit: Int?
    var gapSince: ContinuousClock.Instant?

    init(host: SecureHost, identity: SecureIdentity, endpoint: URL? = nil, invitation: SecureInvitation? = nil,
         trustedObjectOrigin: URL = SecureObjectPolicy.productionOrigin) throws {
        self.host = host; self.identity = identity
        self.invitation = invitation
        self.trustedObjectOrigin = trustedObjectOrigin
        guard let selected = endpoint ?? host.websocketURL else { throw URLError(.cannotFindHost) }
        usesTLS = selected.scheme == "wss"
        var parts = URLComponents(url: selected, resolvingAgainstBaseURL: false)
        guard parts?.user == nil, parts?.password == nil, parts?.fragment == nil,
              parts?.scheme == "wss" || (endpoint != nil && parts?.scheme == "ws") else { throw SecureConnectionError.authenticationFailed }
        let query = (parts?.queryItems ?? []).filter { $0.name != "v" } + [URLQueryItem(name: "v", value: "3")]
        parts?.queryItems = query
        guard let url = parts?.url else { throw SecureConnectionError.protocolViolation }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 15
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.urlCredentialStorage = nil
        urlSession = URLSession(configuration: configuration, delegate: RelayNoRedirect(), delegateQueue: nil)
        let objectConfiguration = URLSessionConfiguration.ephemeral
        objectConfiguration.timeoutIntervalForRequest = 20
        objectConfiguration.timeoutIntervalForResource = 20
        objectConfiguration.urlCache = nil; objectConfiguration.httpCookieStorage = nil; objectConfiguration.urlCredentialStorage = nil
        objectSession = URLSession(configuration: objectConfiguration, delegate: RelayNoRedirect(), delegateQueue: nil)
        socket = urlSession.webSocketTask(with: url)
        socket.maximumMessageSize = 32 * 1024
    }

    static func connect(host: SecureHost, identity: SecureIdentity, endpoint: URL? = nil, timeout: Double = 12,
                        trustedObjectOrigin: URL = SecureObjectPolicy.productionOrigin) async throws -> SecureWebSocketConnection {
        try SecureAdmission.validate(host: host, device: identity.binding())
        let connection = try SecureWebSocketConnection(host: host, identity: identity, endpoint: endpoint, trustedObjectOrigin: trustedObjectOrigin)
        return try await establish(connection, timeout: timeout)
    }

    static func connectInvitation(_ invitation: SecureInvitation, identity: SecureIdentity, endpoint: URL? = nil, timeout: Double = 12) async throws -> SecureWebSocketConnection {
        try invitation.validate()
        guard invitation.invite.expiresAt > UInt64(Date().timeIntervalSince1970) else { throw PairingClaimError.expiredOrUsed }
        let host = SecureHost(binding: invitation.binding, namespace: invitation.namespace, websocketURL: invitation.websocketURL,
            grant: .init(id: "", expiresAt: 0, epoch: 0, signature: ""))
        let connection = try SecureWebSocketConnection(host: host, identity: identity, endpoint: endpoint, invitation: invitation)
        return try await establish(connection, timeout: timeout)
    }

    private static func establish(_ connection: SecureWebSocketConnection, timeout: Double) async throws -> SecureWebSocketConnection {
        do {
            try await withTaskCancellationHandler {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    group.addTask { try await connection.establish() }
                    group.addTask {
                        try await Task.sleep(for: .seconds(timeout))
                        await connection.close()
                        throw URLError(.timedOut)
                    }
                    try await group.next(); group.cancelAll()
                }
            } onCancel: { Task { await connection.close() } }
            return connection
        } catch { await connection.close(); throw error }
    }

    func establish() async throws {
        socket.resume()
        let binding = try identity.binding()
        var hello: [String: Any] = ["a": "hello", "v": 3, "namespace": host.namespace, "hostKey": host.binding.signingKey, "role": invitation == nil ? "device" : "invite"]
        if invitation == nil { hello["objects"] = 1 }
        try await sendJSON(hello)
        let challenge = try await receiveJSON()
        if Self.isGatewayBusy(challenge) { throw URLError(.resourceUnavailable) }
        let now = UInt64(Date().timeIntervalSince1970)
        guard challenge["a"] as? String == "challenge", Self.number(challenge["v"]) == 3,
              let connectionID = challenge["connectionId"] as? String,
              let nonce = challenge["nonce"] as? String, let expires = Self.number(challenge["expiresAt"]),
              expires > now, expires <= now + 60 else { throw SecureConnectionError.authenticationFailed }
        let signingIdentity = try invitation?.invitationIdentity() ?? identity
        let signature = try signingIdentity.sign(SecureAdmission.join(namespace: host.namespace, role: invitation == nil ? 2 : 3, connectionID: connectionID,
                                                               nonce: nonce, device: binding, expiresAt: expires))
        var join: [String: Any] = ["a": "join", "v": 3, "binding": try SecureAdmission.object(binding), "signature": SecureHex.encode(signature)]
        if let invitation { join["invite"] = try SecureAdmission.object(invitation.invite) }
        else { join["grant"] = try SecureAdmission.object(host.grant) }
        try await sendJSON(join)
        var ready: [String: Any]?, accepted: [String: Any]?
        for _ in 0..<4 {
            let frame = try await receiveJSON()
            if Self.isGatewayBusy(frame) { throw URLError(.resourceUnavailable) }
            guard Self.number(frame["v"]) == 3 else { throw SecureConnectionError.protocolViolation }
            if frame["a"] as? String == "ready" { ready = frame }
            else if frame["a"] as? String == "accept" { accepted = frame }
            else { throw SecureConnectionError.authenticationFailed }
            if ready != nil, accepted != nil { break }
        }
        guard let ready, let accepted, let sid = ready["sid"] as? String, let rid = ready["rid"] as? String,
              accepted["sid"] as? String == sid, accepted["rid"] as? String == rid else { throw SecureConnectionError.authenticationFailed }
        self.sid = sid; self.rid = rid
        configureObjectPolicy(ready["objects"])
        let noise = try SecureNoiseSession(identity: identity, namespace: host.namespace, host: host.binding, sid: sid, rid: rid)
        self.noise = noise
        readerTask = Task { await receiveLoop() }
        timerTask = Task { await maintain() }
        try await sendPacket(noise.writeHandshake())
        try noise.readHandshake(await readPacket())
        try await sendPacket(noise.writeHandshake())
        guard noise.ready else { throw SecureConnectionError.authenticationFailed }
    }

    func configureObjectPolicy(_ offered: Any?) {
        if invitation == nil, usesTLS {
            objectPolicy = SecureObjectPolicy.accept(offered, trusted: trustedObjectOrigin)
        }
    }

    func send(_ record: SecureHTTPRecord) async throws {
        guard let noise, noise.ready else { throw failure ?? SecureConnectionError.closed }
        try await sendPacket(noise.seal(record), bulk: record.kind == .data && record.payload.count >= 8192)
    }
    func receive() async throws -> SecureHTTPRecord {
        guard let noise, noise.ready else { throw failure ?? SecureConnectionError.closed }
        return try noise.open(await readPacket())
    }
    func handshakeHash() throws -> Data {
        guard let noise, noise.ready else { throw SecureConnectionError.authenticationFailed }
        return try noise.handshakeHash()
    }

    func sendPacket(_ bytes: Data, bulk: Bool = false) async throws {
        try Task.checkCancellation()
        if let failure { throw failure }
        guard !bytes.isEmpty, bytes.count <= 65_535, sendQueue.count < 64 else { throw SecureConnectionError.capacity }
        var framed = SecureHTTPRecord.u32(UInt32(bytes.count)); framed.append(bytes)
        try await withCheckedThrowingContinuation { continuation in
            sendQueue.append(PacketSend(bytes: framed, bulk: bulk, continuation: continuation))
            if senderTask == nil { senderTask = Task { await drainSends() } }
        }
    }
    func drainSends() async {
        do {
            while !sendQueue.isEmpty {
                let packet = sendQueue.removeFirst(); sending = packet
                let stagePacket = shouldStagePacket(bulk: packet.bulk)
                if !stagePacket {
                    while !stagedFrames.isEmpty {
                        flushObjects()
                        if !stagedFrames.isEmpty { try await withCheckedThrowingContinuation { writer = $0 } }
                        if let failure { throw failure }
                    }
                }
                for offset in stride(from: 0, to: packet.bytes.count, by: 16_384) {
                    while remoteCredit == 0 || pending.count >= SecureRelayWindow.capacity {
                        if let failure { throw failure }
                        try await withCheckedThrowingContinuation { writer = $0 }
                    }
                    if let failure { throw failure }
                    let payload = packet.bytes.subdata(in: offset..<min(offset + 16_384, packet.bytes.count))
                    if stagePacket {
                        while !SecureCipherObject.canAppend(stagedFrames.map(\.bytes), payload) {
                            flushObjects()
                            if !stagedFrames.isEmpty { try await withCheckedThrowingContinuation { writer = $0 } }
                            if let failure { throw failure }
                        }
                    }
                    let bytes = try JSONSerialization.data(withJSONObject: ["a": "data", "v": 3, "sid": sid, "rid": rid,
                        "seq": outgoing, "d": payload.base64EncodedString()])
                    let sequence = outgoing
                    pending[sequence] = PendingFrame(bytes: bytes, sent: .now, tries: 0)
                    outgoing += 1; remoteCredit -= 1
                    if stagePacket {
                        pending[sequence]?.sent = nil
                        stagedFrames.append((sequence, payload))
                        scheduleObjectFlush()
                    } else {
                        flushObjects()
                        try await socket.send(.string(String(decoding: bytes, as: UTF8.self)))
                        inlineSentBytes += payload.count
                    }
                }
                if let failure { throw failure }
                sending = nil; packet.continuation.resume()
            }
            senderTask = nil
        } catch { await fail(error) }
    }

    func readPacket() async throws -> Data {
        if let failure { throw failure }
        if let packet = try takePacket() { try await acknowledge(); return packet }
        guard packetReader == nil else { throw SecureConnectionError.protocolViolation }
        return try await withCheckedThrowingContinuation { packetReader = $0 }
    }
    func takePacket() throws -> Data? {
        packetBytes.append(window.drain(capacity: max(0, 131_078 - packetBytes.count)))
        guard packetBytes.count >= 4 else { return nil }
        let count = packetBytes.prefix(4).reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
        guard count > 0, count <= 65_535 else { throw SecureConnectionError.protocolViolation }
        guard packetBytes.count >= Int(count) + 4 else { return nil }
        let packet = Data(packetBytes.dropFirst(4).prefix(Int(count)))
        packetBytes.removeFirst(Int(count) + 4)
        return packet
    }

    func receiveLoop() async {
        do { while !Task.isCancelled { try await receiveFrame(receiveJSON()) } }
        catch { await fail(error) }
    }
    func receiveFrame(_ frame: [String: Any]) async throws {
        // API Gateway can reject an invocation before Lambda runs. Its envelope
        // identifies no sequence; retain every exact pending ciphertext frame
        // under the existing bounded retry deadline. Never replay an HTTP call.
        if Self.isGatewayBusy(frame) { return }

        guard Self.number(frame["v"]) == 3, let action = frame["a"] as? String else { throw SecureConnectionError.protocolViolation }
        if action == "heartbeat" { return }
        guard frame["sid"] as? String == sid, frame["rid"] as? String == rid else { throw SecureConnectionError.protocolViolation }
        switch action {
        case "object": try receiveObjectReply(frame)
        case "ack":
            guard let seq = Self.number(frame["ack_seq"]), let next = Self.number(frame["next"]),
                  let credit = Self.number(frame["credit"]), credit <= UInt64(SecureRelayWindow.capacity) else { throw SecureConnectionError.protocolViolation }
            if let lastAck, seq <= lastAck { return }
            guard next <= outgoing, next >= acknowledged else { throw SecureConnectionError.protocolViolation }
            acknowledged = next; lastAck = seq
            retireObjects(next: next)
            for key in pending.keys where key < next { pending.removeValue(forKey: key) }
            remoteCredit = max(0, Int(credit) - Int(outgoing - next))
            let waiter = writer; writer = nil; waiter?.resume()
        case "data":
            guard let seq = Self.number(frame["seq"]), (frame["d"] != nil) != (frame["o"] != nil) else { throw SecureConnectionError.protocolViolation }
            if let object = frame["o"] as? [String: Any] {
                if try receiveObject(sequence: seq, value: object) { try await acknowledge(force: true) }
                return
            }
            guard let encoded = frame["d"] as? String,
                  let bytes = Data(base64Encoded: encoded), bytes.base64EncodedString() == encoded else { throw SecureConnectionError.protocolViolation }
            let duplicate = try window.insert(sequence: seq, data: bytes)
            if !duplicate { inlineReceivedBytes += bytes.count }
            packetBytes.append(window.drain(capacity: max(0, 131_078 - packetBytes.count)))
            if let reader = packetReader, let packet = try takePacket() {
                packetReader = nil; reader.resume(returning: packet)
            }
            if window.hasGap { if gapSince == nil { gapSince = .now } } else { gapSince = nil }
            try await acknowledge(force: duplicate)
        case "accept": break
        case "cancel", "eof": throw SecureConnectionError.closed
        default: throw SecureConnectionError.protocolViolation
        }
    }
    func acknowledge(force: Bool = false) async throws {
        if objectPolicy != nil, !force {
            guard ackTask == nil else { return }
            ackTask = Task {
                do { try await Task.sleep(for: .milliseconds(8)); ackTask = nil; try await emitAcknowledgement() }
                catch { if !Task.isCancelled { await fail(error) } }
            }
            return
        }
        try await emitAcknowledgement(force: force)
    }
    func emitAcknowledgement(force: Bool = false) async throws {
        let credit = packetBytes.count >= 65_539 ? 0 : window.credit
        if force, advertisedNext == window.next, advertisedCredit == credit, let lastSentAcknowledgement {
            try await socket.send(.string(String(decoding: lastSentAcknowledgement, as: UTF8.self)))
            return
        }
        guard force || advertisedNext != window.next || advertisedCredit != credit else { return }
        advertisedNext = window.next; advertisedCredit = credit
        let seq = ackSequence; ackSequence += 1
        let bytes = try JSONSerialization.data(withJSONObject: ["a": "ack", "v": 3, "sid": sid, "rid": rid, "next": window.next, "credit": credit, "ack_seq": seq])
        lastSentAcknowledgement = bytes
        try await socket.send(.string(String(decoding: bytes, as: UTF8.self)))
    }
    func maintain() async {
        var ticks = 0
        do {
            while !Task.isCancelled {
                try await Task.sleep(for: .seconds(10)); ticks += 1
                if let gapSince, gapSince.duration(to: .now) >= .seconds(objectPolicy == nil ? 30 : 90) { throw SecureConnectionError.protocolViolation }
                var repeated = Set<Data>()
                for (seq, value) in pending {
                    guard let sent = value.sent, sent.duration(to: .now) >= .seconds(10) else { continue }
                    if let expiry = value.expiresAt, expiry <= UInt64(Date().timeIntervalSince1970) { throw URLError(.timedOut) }
                    guard value.tries < 3 else { throw URLError(.timedOut) }
                    pending[seq]?.tries += 1; pending[seq]?.sent = .now
                    if repeated.insert(value.bytes).inserted { try await socket.send(.string(String(decoding: value.bytes, as: UTF8.self))) }
                }
                if ticks % 3 == 0 { try await sendJSON(["a": "heartbeat", "v": 3]) }
            }
        } catch { await fail(error) }
    }
    func close() async { await fail(SecureConnectionError.closed) }
    func fail(_ error: any Error) async {
        guard failure == nil else { return }
        failure = error
        socket.cancel(with: .goingAway, reason: nil); urlSession.invalidateAndCancel()
        objectSession.invalidateAndCancel()
        readerTask?.cancel(); timerTask?.cancel(); senderTask?.cancel()
        objectFlushTask?.cancel(); ackTask?.cancel()
        objectFlushTask = nil; ackTask = nil
        for task in objectSends.values { task.cancel() }
        for fetch in objectFetches.values { fetch.task.cancel() }
        for request in objectRequests.values { request.timeout.cancel(); request.continuation.resume(throwing: error) }
        objectSends.removeAll(); objectReservations.removeAll(); objectFetches.removeAll(); objectRequests.removeAll(); completedObjectRequests.removeAll(); stagedFrames.removeAll()
        readerTask = nil; timerTask = nil; senderTask = nil
        writer?.resume(throwing: error); writer = nil
        packetReader?.resume(throwing: error); packetReader = nil
        sending?.continuation.resume(throwing: error); sending = nil
        for packet in sendQueue { packet.continuation.resume(throwing: error) }
        sendQueue.removeAll(); pending.removeAll(); packetBytes = Data(); noise = nil
    }
    func sendJSON(_ object: [String: Any]) async throws { try await socket.send(.string(String(decoding: JSONSerialization.data(withJSONObject: object), as: UTF8.self))) }
    func receiveJSON() async throws -> [String: Any] {
        let message = try await socket.receive()
        let data: Data
        switch message { case let .data(value): data = value; case let .string(value): data = Data(value.utf8); @unknown default: throw SecureConnectionError.protocolViolation }
        guard data.count <= 32 * 1024, let value = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw SecureConnectionError.protocolViolation }
        return value
    }
    static func isGatewayBusy(_ frame: [String: Any]) -> Bool {
        guard Set(frame.keys) == Set(["connectionId", "requestId", "message"]),
              frame["message"] as? String == "Internal server error" else { return false }
        return ["connectionId", "requestId"].allSatisfy { key in
            guard let value = frame[key] as? String else { return false }
            return (1...128).contains(value.utf8.count) && value.utf8.allSatisfy { (33...126).contains($0) }
        }
    }
    static func number(_ value: Any?) -> UInt64? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue >= 0, number.doubleValue <= 9_007_199_254_740_991,
              number.doubleValue.rounded() == number.doubleValue else { return nil }
        return number.uint64Value
    }
}
