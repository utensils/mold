import Foundation

// How it was run, what workflow it belongs to, and the file itself.
extension PrintDetails {

    /// Several clips, and whether a person asked for them.
    ///
    /// `chain` rides an authored sequence AND a one shot the host had to
    /// render in pieces because the model cannot do that length in one pass;
    /// `output_mode` is what tells them apart (`chain.rs:66-70`, and the
    /// repository's own note on ephemeral chains). Saying "sequence" over an
    /// automatic one would credit somebody with authoring a split they never
    /// made.
    static func sequenceGroup(_ meta: OutputMetadata) -> PrintDetailGroup? {
        guard meta.chain != nil || meta.chainJobId != nil else { return nil }
        let authored = meta.outputMode == "sequence"
        let clips = meta.chain?.stageCount.map { count($0, of: "clip") }
        return group(authored ? "Sequence" : "Rendered in clips", [
            row("Made of", clips.map { authored ? "\($0), one prompt each" : $0 }),
            row("Motion tail",
                meta.chain.flatMap(\.motionTailFrames).map { count($0, of: "frame") }),
            row("Sequence job", meta.chainJobId),
        ])
    }

    /// The durable 3-D run a stage belongs to. Every stage of a workflow
    /// publishes an ORDINARY print, so without this a text-to-3-D run reads as
    /// four unrelated prints (`mesh_workflow.rs:176-190`).
    static func workflowGroup(_ meta: OutputMetadata) -> PrintDetailGroup? {
        guard let workflow = meta.meshWorkflow else { return nil }
        return group("3-D workflow", [
            row("Workflow", workflow.mode.map(spelledOut)),
            row("Step", workflow.role.map(spelledOut)),
            row("Workflow job", workflow.jobId),
        ])
    }

    /// The vocabulary the host sends, in words. A value this build has never
    /// heard of is still shown -- opened out rather than dropped, because a
    /// newer host naming a new stage is not an error.
    static func spelledOut(_ wire: String) -> String {
        switch wire {
        case "text_to_mesh": "Text to 3-D"
        case "mesh_roundtrip": "Mesh round trip"
        case "mesh_texture": "Texture a mesh"
        case "generated_image": "The picture it started from"
        case "matted_image": "Background removed"
        case "delighted_image": "Lighting removed"
        case "final_glb": "The finished mesh"
        default: wire.replacingOccurrences(of: "_", with: " ")
        }
    }

    static func fileGroup(_ entry: LibraryEntry) -> PrintDetailGroup? {
        let print = entry.print
        let meta = print.metadata
        return group("File", [
            row("Name", print.filename),
            row("Format", (print.format ?? meta.outputFormat)?.uppercased()),
            row("Size", size(meta.width, meta.height)),
            // Only when the file is not the canvas: an upscaled print's own
            // size is the big one, and the number to reuse is this.
            rendered(meta),
            row("On disk", print.sizeBytes.map { Int64($0).formatted(.byteCount(style: .file)) }),
            row("Made", entry.createdAt.formatted(date: .abbreviated, time: .shortened)),
            row("Took", took(meta.generationTimeMs)),
            row("Job", meta.jobId),
            row("mold", meta.version),
        ])
    }

    private static func rendered(_ meta: OutputMetadata) -> PrintDetailRow? {
        guard let canvas = size(meta.generationWidth, meta.generationHeight),
              canvas != size(meta.width, meta.height) else { return nil }
        return row("Rendered at", canvas)
    }

    static func size(_ width: Int?, _ height: Int?) -> String? {
        guard let width, let height else { return nil }
        return "\(width.formatted()) × \(height.formatted())"
    }

    /// Wall clock, as a person says it. Zero is the wire's "not measured"
    /// (`types.rs:3415-3417`) and leaves the row out.
    static func took(_ milliseconds: Int?) -> String? {
        guard let milliseconds, milliseconds > 0 else { return nil }
        let seconds = Double(milliseconds) / 1_000
        guard seconds >= 60 else {
            return "\(seconds.formatted(.number.precision(.fractionLength(0 ... 1)))) sec"
        }
        let whole = Int(seconds.rounded())
        let rest = whole % 60
        return rest == 0 ? "\(whole / 60) min" : "\(whole / 60) min \(rest) sec"
    }
}
