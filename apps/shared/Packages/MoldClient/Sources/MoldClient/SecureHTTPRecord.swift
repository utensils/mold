import Foundation

/// The authenticated HTTP record format owned by `mold-secure::records`.
/// This codec describes the envelope; Rust authenticates and validates every received record.
struct SecureHTTPRecord: Equatable, Sendable {
    static let streamWindow = 1_048_576
    static let creditBatch = 524_288
    enum Kind: UInt8, Sendable { case open = 1, headers, data, end, cancel, credit }
    let kind: Kind
    let streamID: UInt32
    let payload: Data

    init(_ kind: Kind, _ streamID: UInt32, _ payload: Data = Data()) {
        self.kind = kind; self.streamID = streamID; self.payload = payload
    }

    init(encoded data: Data) throws {
        guard data.count >= 9, data.count <= 65_519,
              let kind = Kind(rawValue: data[data.startIndex]) else { throw SecureConnectionError.protocolViolation }
        let bytes = Array(data.prefix(9))
        let id = bytes[1...4].reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
        let length = bytes[5...8].reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
        guard id != 0, length == data.count - 9 else { throw SecureConnectionError.protocolViolation }
        self.init(kind, id, Data(data.dropFirst(9)))
        try validate()
    }

    func encoded() throws -> Data {
        try validate()
        var result = Data([kind.rawValue])
        result.append(Self.u32(streamID)); result.append(Self.u32(UInt32(payload.count))); result.append(payload)
        return result
    }

    private func validate() throws {
        guard streamID != 0, payload.count <= 65_510 else { throw SecureConnectionError.protocolViolation }
        switch kind {
        case .data: guard !payload.isEmpty, payload.count <= 16_384 else { throw SecureConnectionError.protocolViolation }
        case .end: guard payload.isEmpty else { throw SecureConnectionError.protocolViolation }
        case .credit: guard let value = credit, (1...UInt32(Self.streamWindow)).contains(value) else { throw SecureConnectionError.protocolViolation }
        case .cancel: guard payload.count <= 64, payload.allSatisfy({ $0 < 128 }) else { throw SecureConnectionError.protocolViolation }
        case .open, .headers: break
        }
    }

    var credit: UInt32? {
        guard kind == .credit, payload.count == 4 else { return nil }
        return payload.reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
    }

    static func u32(_ value: UInt32) -> Data {
        Data([UInt8(truncatingIfNeeded: value >> 24), UInt8(truncatingIfNeeded: value >> 16),
              UInt8(truncatingIfNeeded: value >> 8), UInt8(truncatingIfNeeded: value)])
    }
}

protocol SecureRecordConnection: Sendable {
    func send(_ record: SecureHTTPRecord) async throws
    func receive() async throws -> SecureHTTPRecord
    func close() async
}
