import Foundation

/// Starting and cancelling a model fetch, and reading the machine's whole
/// download queue.
public protocol MoldDownloadsBackend: Sendable {
    func startDownload(_ request: DownloadRequest) async throws -> DownloadTicket
    func cancelDownload(id: String) async throws
    /// Every job this machine is running, queued or has finished, whoever
    /// asked for it (`types.rs:13274-13285`).
    func downloads() async throws -> DownloadsListing
}
