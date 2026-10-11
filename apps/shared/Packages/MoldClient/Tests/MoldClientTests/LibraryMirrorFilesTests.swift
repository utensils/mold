import CryptoKit
import Foundation
import Testing
@testable import MoldClient

struct LibraryMirrorFilesTests {
    @Test @MainActor func blockingWorkerLeavesMainThread() async throws {
        let onMain = try await LibraryMirrorFiles.run { Thread.isMainThread }
        #expect(!onMain)
    }

    @Test func comparisonAndHashStreamPastFirstChunk() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = directory.appendingPathComponent("first"), second = directory.appendingPathComponent("second")
        var data = Data(repeating: 70, count: 2 * 1_024 * 1_024 + 17)
        try data.write(to: first); try data.write(to: second)
        #expect(try await LibraryMirrorFiles.equal(first, second))
        let expected = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(try await LibraryMirrorFiles.sha256(first) == expected)
        data[data.count - 1] = 71
        try data.write(to: second)
        #expect(try await !LibraryMirrorFiles.equal(first, second))
        try Data([1]).write(to: second)
        #expect(try await !LibraryMirrorFiles.equal(first, second))
    }

    @Test func mirrorDescriptorRemainsFileBackedAndPreservesRecipe() async throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data([1, 2, 3]).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"source.png","timestamp":123,"metadata":{"prompt":"A source","model":"fixture"}}"#.utf8))
        let item = try await LibraryMirrorFiles.importDescriptor(for: print, file: file)
        #expect(item.file.isEmpty)
        #expect(item.fileURL == file)
        #expect(item.prompt == print.metadata.prompt ?? "")
        #expect(item.timestamp == print.createdAt)
        #expect(item.originalMetadataJSON == print.rawMetadataJSON)
    }

    @Test func cancelledOperationsDoNotReadMissingFiles() async throws {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await LibraryMirrorFiles.sha256(missing)
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    @Test func workerPropagatesCancellationDuringWork() async throws {
        let (started, signal) = AsyncStream<Void>.makeStream()
        let task = Task {
            try await LibraryMirrorFiles.run {
                signal.yield(())
                while true {
                    try Task.checkCancellation()
                    Thread.sleep(forTimeInterval: 0.001)
                }
            }
        }
        for await _ in started { break }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
    }
}
