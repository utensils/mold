import Foundation

/// Making a print bigger: one synchronous call for a still, a durable job for
/// a clip.
///
/// Both are gated on `capabilities.video_upscale` and nothing else -- a still
/// additionally needs `gallery_image`, because the server-side publication
/// endpoint is newer than the block that advertises it
/// (`types.rs:12430-12433`).
public protocol MoldUpscaleBackend: Sendable {
    /// `POST /api/gallery/upscale`. Synchronous: the bigger still is written
    /// into this machine's Library before the call returns, so there is
    /// nothing to follow afterwards except a gallery re-read.
    func upscaleLibraryImage(
        filename: String, model: String, tileSize: Int?
    ) async throws -> GalleryImageUpscale

    /// `POST /api/video-upscale-jobs`. Answers `202` with the new job.
    func startFramewiseUpscale(
        filename: String, model: String, tileSize: Int?
    ) async throws -> VideoUpscaleJob

    /// Every durable clip upscale this machine remembers, settled ones
    /// included -- how a job is recovered onto the print it belongs to.
    func framewiseUpscales() async throws -> [VideoUpscaleJob]

    func framewiseUpscale(id: String) async throws -> VideoUpscaleJob

    /// Pause, resume, or cancel. Each answers the job in its new state, so a
    /// caller never has to re-read to find out what it did.
    func transitionFramewiseUpscale(
        id: String, to transition: FramewiseTransition
    ) async throws -> VideoUpscaleJob
}
