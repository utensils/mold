import Foundation

// The upscale routes. Their own file rather than more of `HTTPBackend`,
// which is already past the type-size budget.
public extension HTTPBackend {

    /// A still, upscaled and published in one call (`routes.rs:853-856`).
    ///
    /// The timeout is the generous one: this runs the model inline on the
    /// request thread, and a 4x pass over a 1024px still on a busy machine
    /// outlasts the ordinary 10 s idle limit -- the same reason a cold
    /// prompt expansion asks for more.
    func upscaleLibraryImage(
        filename: String, model: String, tileSize: Int?
    ) async throws -> GalleryImageUpscale {
        try await post("/api/gallery/upscale",
                       body: GalleryUpscaleBody(filename: filename, model: model,
                                                tileSize: tileSize),
                       timeout: 300)
    }

    func startFramewiseUpscale(
        filename: String, model: String, tileSize: Int?
    ) async throws -> VideoUpscaleJob {
        try await post("/api/video-upscale-jobs",
                       body: FramewiseUpscaleBody(source: .library(filename: filename),
                                                  model: model, tileSize: tileSize))
    }

    func framewiseUpscales() async throws -> [VideoUpscaleJob] {
        try await get("/api/video-upscale-jobs")
    }

    func framewiseUpscale(id: String) async throws -> VideoUpscaleJob {
        try await get("/api/video-upscale-jobs/\(escaped(id))")
    }

    /// Cancel is `DELETE` on the job itself; pause and resume are POSTs to a
    /// sub-path (`routes.rs:861-875`). All three answer the job in its NEW
    /// state, so nothing has to re-read to find out what it did.
    ///
    /// Written as three whole literals rather than one built by
    /// concatenation, because both route contract tests scan for a string
    /// beginning `/api` -- a path assembled out of pieces is a path neither
    /// of them can see.
    func transitionFramewiseUpscale(
        id: String, to transition: FramewiseTransition
    ) async throws -> VideoUpscaleJob {
        switch transition {
        case .cancel:
            try await transitioned("/api/video-upscale-jobs/\(escaped(id))", method: "DELETE")
        case .pause:
            try await transitioned("/api/video-upscale-jobs/\(escaped(id))/pause", method: "POST")
        case .resume:
            try await transitioned("/api/video-upscale-jobs/\(escaped(id))/resume", method: "POST")
        }
    }

    private func transitioned(_ path: String, method: String) async throws -> VideoUpscaleJob {
        let data = try await bytes(for: request(path, method: method))
        return try decoded(VideoUpscaleJob.self, from: data, route: path)
    }
}

/// `tile_size` is `#[serde(default)]` on the server, so an absent key is a
/// real "the host chooses" -- never `null`, and never a number this app made
/// up (`video_upscale.rs:670-676`).
private struct GalleryUpscaleBody: Encodable {
    let filename: String
    let model: String
    let tileSize: Int?
}

private struct FramewiseUpscaleBody: Encodable {
    let source: VideoUpscaleSource
    let model: String
    let tileSize: Int?
}
