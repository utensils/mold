import Foundation

/// What a machine says it can make bigger.
///
/// `canUpscaleClips` is already `Capabilities+Reading`'s; this is the other
/// half. Its own file rather than a line there, because the block advertises
/// TWO different answers and only one of them was ever read.
public extension Capabilities {
    /// Whether this machine can upscale a still IN ITS OWN LIBRARY.
    ///
    /// `gallery_image` is newer than the block that carries it, so a host
    /// advertising `video_upscale` without it can upscale clips and not
    /// stills -- `"Absent/false on older hosts, whose clients must retain the
    /// legacy upscale-stream fallback"` (`types.rs:12430-12433`). This app has
    /// no such fallback: it offers the action where the host publishes the
    /// result, and nowhere else.
    var canUpscaleStills: Bool {
        guard let block = videoUpscale, block.available else { return false }
        return block.galleryImage == true
    }

    /// The host's own caveat about what framewise upscaling does to a clip,
    /// shown verbatim wherever the action is offered.
    var framewiseDisclosure: String? { videoUpscale?.disclosure }
}
