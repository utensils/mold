import Foundation

/// The machine's live event feeds.
public protocol MoldStreamsBackend: Sendable {
    /// Everything this machine reports, as it happens.
    func events() -> AsyncThrowingStream<MoldEvent, Error>
    /// Whole-snapshot frames; safe to reconnect at any point.
    func batchEvents(id: String) -> AsyncThrowingStream<BatchStatus, Error>
    /// Progress for every download on this host.
    func downloadEvents() -> AsyncThrowingStream<DownloadEvent, Error>
}
