import Foundation

// What the print was made FROM, and what kind of thing it is.
extension PrintDetails {

    /// A clip's own facts. Read off the PRINT rather than the metadata alone,
    /// because `isVideo` is the one answer for what moves -- a WebP is a still
    /// or a clip depending on its frame count.
    static func clipGroup(_ print: GalleryPrint) -> PrintDetailGroup? {
        guard print.isVideo else { return nil }
        let meta = print.metadata
        return group("Clip", [
            row("Frames", meta.frames.map { $0.formatted() }),
            row("Rate", meta.fps.map { "\(decimal($0, places: 2) ?? "") fps" }),
            row("Length", length(frames: meta.frames, fps: meta.fps)),
            row("Audio", meta.enableAudio.map { $0 ? "On" : "Off" }),
            // What RAN, not what was asked for: `pipelineRequested` is the
            // separate bit that says whether the request named it.
            row("Pipeline", meta.pipeline),
            row("Continues", meta.extendOverlapFrames.map {
                "the last \(count($0, of: "frame")) of another clip"
            }),
        ])
    }

    static func length(frames: Int?, fps: Double?) -> String? {
        guard let frames, let fps, fps > 0 else { return nil }
        let seconds = Double(frames) / fps
        return "\(seconds.formatted(.number.precision(.fractionLength(0 ... 1)))) sec"
    }

    /// The 3-D controls that RAN -- resolved, not requested, so a print made
    /// with the recipe's own defaults still says what they were.
    static func meshGroup(_ meta: OutputMetadata) -> PrintDetailGroup? {
        guard let mesh = meta.mesh else { return nil }
        return group("3-D", [
            row("Detail", mesh.octreeResolution.map { "\($0.formatted())³ grid" }),
            row("Iso threshold", decimal(mesh.threshold, places: 2)),
            row("Faces", mesh.targetFaces.map { $0.formatted() }),
            row("Texture", mesh.texture.map { $0 ? "On" : "Off" }),
        ])
    }

    /// Labels and counts only. mold records the NAME and the digest of what
    /// conditioned a render and never the pixels, so this is everything there
    /// is to show.
    static func sourcesGroup(_ meta: OutputMetadata) -> PrintDetailGroup? {
        let photos = meta.identityPhotoNames
        return group("Made from", [
            row("Source picture", meta.sourceImageName),
            // Digests only ever answer HOW MANY -- mold records names and
            // hashes for conditioning, never the bytes.
            row("References", references(meta)),
            row("Keyframes", meta.keyframes.map { count($0.count, of: "still") }),
            row(photos.count == 1 ? "Identity photo" : "Identity photos",
                photos.isEmpty ? nil : photos.joined(separator: ", ")),
            row("Identity strength", decimal(meta.idWeight, places: 2)),
            row("Identity from step", meta.idStartStep.map(String.init)),
            row("Control", meta.controlModel),
            row("Control strength", decimal(meta.controlScale, places: 2)),
            row(adapters(meta).count == 1 ? "Adapter" : "Adapters",
                adapters(meta).isEmpty ? nil : adapters(meta).joined(separator: ", ")),
            row("Made bigger with", meta.upscaleModel),
        ])
    }

    /// An H3 print names its references; every other family records only
    /// their digests, so the honest answer there is a count.
    static func references(_ meta: OutputMetadata) -> String? {
        if let named = meta.references, !named.isEmpty {
            let labels = named.compactMap(\.name)
            return labels.count == named.count ? labels.joined(separator: ", ")
                : count(named.count, of: "picture")
        }
        let digests = meta.editImageDigests
        return digests.isEmpty ? nil : count(digests.count, of: "picture")
    }

    /// The stack as the print recorded it, newer `loras` first and the single
    /// legacy `lora` as the fallback -- the same precedence the server's own
    /// metadata builder uses (`types.rs:3425-3429`). The wire carries a
    /// server-side PATH and no name, and the file's own name is the closest
    /// thing to one a reader would recognise.
    static func adapters(_ meta: OutputMetadata) -> [String] {
        if let loras = meta.loras, !loras.isEmpty {
            return loras.map { adapter(name(of: $0.path), scale: $0.scale) }
        }
        guard let lora = meta.lora else { return [] }
        return [adapter(name(of: lora), scale: meta.loraScale)]
    }

    /// Never a path: the rest of it is a stranger's directory layout.
    static func name(of path: String) -> String { (path as NSString).lastPathComponent }

    private static func adapter(_ name: String, scale: Double?) -> String {
        guard let scale = decimal(scale, places: 2) else { return name }
        return "\(name) at \(scale)"
    }

    static func count(_ number: Int, of noun: String) -> String {
        "\(number.formatted()) \(number == 1 ? noun : noun + "s")"
    }
}
