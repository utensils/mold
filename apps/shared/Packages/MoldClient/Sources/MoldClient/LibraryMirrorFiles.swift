import CryptoKit
import Foundation

/// Blocking mirror-file work always leaves the caller's actor. Callers retain
/// their downloaded files until these operations finish, including cancellation.
public enum LibraryMirrorFiles {
    private static let chunkSize = 1_024 * 1_024

    public static func equal(_ first: URL, _ second: URL) async throws -> Bool {
        try await run {
            let manager = FileManager.default
            let firstSize = try manager.attributesOfItem(atPath: first.path)[.size] as? NSNumber
            let secondSize = try manager.attributesOfItem(atPath: second.path)[.size] as? NSNumber
            guard firstSize == secondSize else { return false }
            let left = try FileHandle(forReadingFrom: first)
            defer { try? left.close() }
            let right = try FileHandle(forReadingFrom: second)
            defer { try? right.close() }
            while true {
                try Task.checkCancellation()
                let a = try left.read(upToCount: chunkSize) ?? Data()
                try Task.checkCancellation()
                let b = try right.read(upToCount: chunkSize) ?? Data()
                guard a == b else { return false }
                if a.isEmpty { return true }
            }
        }
    }

    public static func sha256(_ file: URL) async throws -> String {
        try await run {
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            var digest = SHA256()
            while true {
                try Task.checkCancellation()
                guard let chunk = try handle.read(upToCount: chunkSize), !chunk.isEmpty else { break }
                digest.update(data: chunk)
            }
            return digest.finalize().map { String(format: "%02x", $0) }.joined()
        }
    }

    public static func importDescriptor(for print: GalleryPrint, file: URL) async throws -> GalleryImport {
        try await run { try GalleryImport(mirroring: print, fileAt: file) }
    }

    static func run<T: Sendable>(_ operation: @escaping @Sendable () throws -> T) async throws -> T {
        try Task.checkCancellation()
        let worker = Task.detached(priority: .utility) {
            try Task.checkCancellation()
            let result = try operation()
            try Task.checkCancellation()
            return result
        }
        return try await withTaskCancellationHandler {
            let result = try await worker.value
            try Task.checkCancellation()
            return result
        } onCancel: { worker.cancel() }
    }
}
