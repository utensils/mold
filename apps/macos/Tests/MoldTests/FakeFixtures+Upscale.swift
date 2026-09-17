import Foundation
import MoldClient

// Lane F3's fixtures. Their own file rather than more of `FakeFixtures.swift`,
// which several lanes are editing at once -- every value here is still built
// the one sanctioned way, by decoding the JSON a host actually sends.
extension FakeFixtures {

    /// `capabilities.video_upscale`. `available` gates clips; `gallery_image`
    /// is the newer field that additionally gates a still, and a host with
    /// the block but not the field is a real shape on the wire
    /// (`types.rs:12430-12433`). `videoUpscale: false` omits the block
    /// entirely, which is what an older host sends.
    static func capabilities(videoUpscale: Bool, galleryImage: Bool = true) -> Capabilities {
        let json = videoUpscale
            ? #"""
            {"video_upscale": {"available": true, "gallery_image": \#(galleryImage),
             "contract_version": 1, "source_library": true, "source_upload": false,
             "disclosure": "Framewise upscale processes each frame independently; temporal flicker may remain."}}
            """#
            : "{}"
        return try! MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    /// An upscaler as `/api/models` lists one. The family is what
    /// `Model.isUpscaler` reads, and `downloaded` is what the default-picking
    /// policy prefers.
    static func upscaler(_ name: String, downloaded: Bool) -> Model {
        model(name, family: "upscaler", downloaded: downloaded)
    }

    /// `POST /api/gallery/upscale`'s answer.
    static func stillUpscale(_ filename: String, model: String = "real-esrgan-x4plus:fp16")
        -> GalleryImageUpscale {
        let json = #"""
        {"filename": "\#(filename)", "model": "\#(model)", "scale_factor": 4}
        """#
        return try! MoldJSON.decoder.decode(GalleryImageUpscale.self, from: Data(json.utf8))
    }

    /// One durable clip upscale, as `/api/video-upscale-jobs` reports it.
    static func framewiseJob(
        _ id: String, state: String, done: Int = 0, total: Int = 0,
        filename: String = "clip.mp4", error: String? = nil
    ) -> VideoUpscaleJob {
        let json = #"""
        {"id": "\#(id)", "state": "\#(state)", "model": "real-esrgan-x4plus:fp16",
         "completed_frames": \#(done), "total_frames": \#(total),
         "error": \#(error.map { "\"\($0)\"" } ?? "null"),
         "source": {"kind": "library", "filename": "\#(filename)"}}
        """#
        return try! MoldJSON.decoder.decode(VideoUpscaleJob.self, from: Data(json.utf8))
    }
}
