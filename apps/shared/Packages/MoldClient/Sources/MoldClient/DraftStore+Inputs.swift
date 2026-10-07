import CryptoKit
import Foundation

extension DraftStore {
    public static let maximumInputSnapshotBytes = ResponseCeiling.media * 2

    /// Read only the digest named by the same saved descriptor. A missing or
    /// corrupt attachment snapshot is an error, never an empty input set.
    public func loadInputs(for descriptor: DraftDescriptor) throws -> DraftInputSnapshot? {
        guard let digest = descriptor.localInputsSHA256 else { return nil }
        guard RetainedSourceMedia.validTransferDigest(digest) else { throw CocoaError(.fileReadCorruptFile) }
        let file = inputsDirectory.appending(path: "\(digest).json")
        guard let stamp = Self.stamp(file), stamp.size <= Self.maximumInputSnapshotBytes else {
            throw CocoaError(.fileReadTooLarge)
        }
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var bytes = Data()
        while let chunk = try handle.read(upToCount: 1_024 * 1_024), !chunk.isEmpty {
            guard bytes.count <= stamp.size - chunk.count else { throw CocoaError(.fileReadCorruptFile) }
            bytes.append(chunk)
        }
        guard bytes.count == stamp.size else { throw CocoaError(.fileReadCorruptFile) }
        guard Self.digest(bytes) == digest else { throw CocoaError(.fileReadCorruptFile) }
        let snapshot = try MoldJSON.localDecoder.decode(DraftInputSnapshot.self, from: bytes)
        var checked = DraftMedia()
        try snapshot.apply(to: &checked)
        return snapshot
    }

    var inputsDirectory: URL { url.deletingLastPathComponent().appending(path: "generate-draft-inputs", directoryHint: .isDirectory) }

    func saveInputs(_ snapshot: DraftInputSnapshot) throws -> String {
        // Prompt keystrokes do not re-encode or rewrite unchanged image/video
        // payloads. The snapshot's own authoring equality controls this cache.
        if let cached = writes.inputs, cached.snapshot == snapshot,
           let stamp = Self.stamp(inputsDirectory.appending(path: "\(cached.digest).json")), stamp == cached.stamp {
            return cached.digest
        }
        let bytes = try MoldJSON.localEncoder.encode(snapshot)
        guard bytes.count <= Self.maximumInputSnapshotBytes else { throw CocoaError(.fileWriteOutOfSpace) }
        let digest = Self.digest(bytes)
        try FileManager.default.createDirectory(at: inputsDirectory, withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
        let file = inputsDirectory.appending(path: "\(digest).json")
        try writePrivate(bytes, to: file)
        if let stamp = Self.stamp(file) { writes.inputs = (snapshot, digest, stamp) }
        return digest
    }

    func pruneInputs(keeping digests: Set<String>) {
        guard let files = try? FileManager.default.contentsOfDirectory(at: inputsDirectory, includingPropertiesForKeys: nil) else { return }
        for file in files where file.pathExtension == "json" {
            let digest = file.deletingPathExtension().lastPathComponent
            if RetainedSourceMedia.validTransferDigest(digest), !digests.contains(digest) {
                try? FileManager.default.removeItem(at: file)
            }
        }
    }

    func writePrivate(_ bytes: Data, to destination: URL) throws {
        let temporary = destination.deletingLastPathComponent().appending(path: "draft-\(UUID().uuidString).tmp")
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard FileManager.default.createFile(atPath: temporary.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
            throw CocoaError(.fileWriteUnknown)
        }
        let handle = try FileHandle(forWritingTo: temporary)
        do { try handle.write(contentsOf: bytes); try handle.synchronize(); try handle.close() }
        catch { try? handle.close(); throw error }
        guard rename(temporary.path, destination.path) == 0 else { throw CocoaError(.fileWriteUnknown) }
    }

    private static func stamp(_ file: URL) -> DraftInputFileStamp? {
        guard let values = try? FileManager.default.attributesOfItem(atPath: file.path),
              values[.type] as? FileAttributeType == .typeRegular,
              let size = values[.size] as? NSNumber,
              let modified = values[.modificationDate] as? Date,
              let inode = values[.systemFileNumber] as? NSNumber else { return nil }
        return DraftInputFileStamp(size: size.intValue, modified: modified, inode: inode.uint64Value)
    }

    private static func digest(_ bytes: Data) -> String {
        SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
}

/// Copies of one DraftStore share serialization and reservation state. Reserving
/// the quit write supersedes an older detached save before it can commit.
final class DraftWriteGate: @unchecked Sendable {
    let writer = NSLock()
    private let reservations = NSLock()
    private var latest: UInt64 = 0
    var inputs: (snapshot: DraftInputSnapshot, digest: String, stamp: DraftInputFileStamp)?

    func reserve() -> UInt64 { reservations.withLock { latest &+= 1; return latest } }
    func isCurrent(_ revision: UInt64) -> Bool { reservations.withLock { latest == revision } }
    func commit(_ revision: UInt64, action: () throws -> Void) throws -> Bool {
        try reservations.withLock {
            guard latest == revision else { return false }
            try action()
            return true
        }
    }
}

struct DraftInputFileStamp: Equatable, Sendable {
    let size: Int
    let modified: Date
    let inode: UInt64
}
