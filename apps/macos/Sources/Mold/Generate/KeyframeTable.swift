import AppKit
import MoldClient
import MoldStyle
import SwiftUI
import UniformTypeIdentifiers

/// One row per keyframe: a frame number and a `MediaWell`.
///
/// Rows stay sorted by frame (`DraftMedia.addingKeyframe`). A frame at or
/// past the clip's own length is never reachable in the first place --
/// `snappedFrame` clamps it the same way `TemporalProfile.snap` clamps a
/// frame COUNT, rather than accepting an out-of-range value and failing
/// later at the wire (`validation.rs:1705-1711`).
struct KeyframeTable: View {
    let temporal: TemporalProfile?
    @Binding var draft: RenderDraft

    /// What a file the engine cannot read said, beside the control that
    /// collected it rather than nowhere at all -- the same caption its two
    /// sibling wells draw (finding 02#7).
    @State private var importFailure: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(draft.media.keyframes.enumerated()), id: \.offset) { index, keyframe in
                row(index: index, keyframe: keyframe)
            }
            addButton
            if let importFailure {
                Text(importFailure).font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private func row(index: Int, keyframe: KeyframeCondition) -> some View {
        if let temporal {
            HStack(spacing: 6) {
                Stepper(value: frameBinding(index: index, temporal: temporal),
                        in: 0...ceiling, step: Swift.max(temporal.frames.step, 1)) {
                    Text(keyframe.frame.formatted())
                        .monospacedDigit()
                        .frame(minWidth: 30, alignment: .trailing)
                }
                MediaWell(
                    systemImage: "photo", allowedTypes: [.png, .jpeg],
                    placeholder: "No picture", attachment: attachmentBinding(index: index)
                )
                Button {
                    draft.media.keyframes.remove(at: index)
                } label: {
                    Image(systemName: "minus.circle")
                }
                .buttonStyle(.plain)
                .help("Remove this keyframe")
            }
        }
    }

    @ViewBuilder private var addButton: some View {
        if let temporal {
            Button {
                addKeyframe(temporal: temporal)
            } label: {
                Label("Add keyframe", systemImage: "plus")
            }
            .buttonStyle(.plain)
            .font(.caption)
        }
    }

    /// Through `PictureSource`'s one panel and `PictureImport` like every
    /// other still: read and encoded off the main actor, and conformed to
    /// something the engine decodes rather than uploaded and refused (findings
    /// 02#7 and 02#10). Its own panel offered PNG and JPEG alone, so an iPhone
    /// photograph could not be picked as a keyframe at all. The frame is
    /// chosen when the bytes arrive, so two picks in a row cannot land on
    /// the same one.
    private func addKeyframe(temporal: TemporalProfile) {
        guard let url = PictureSource.choose().first else { return }
        Task {
            do {
                let picked = try await PictureImport.load(
                    url, accepting: PictureImport.engineReadable)
                draft.media.addingKeyframe(KeyframeCondition(
                    frame: nextFrame(temporal: temporal), image: picked.encoded,
                    name: picked.name))
                importFailure = nil
            } catch {
                importFailure = error.reasonSentence
            }
        }
    }

    private func nextFrame(temporal: TemporalProfile) -> Int {
        guard let last = draft.media.keyframes.map(\.frame).max() else { return 0 }
        return Self.snappedFrame(last + Swift.max(temporal.frames.step, 1), temporal: temporal, frames: clipFrames)
    }

    private func frameBinding(index: Int, temporal: TemporalProfile) -> Binding<Int> {
        Binding(
            get: { draft.media.keyframes.indices.contains(index) ? draft.media.keyframes[index].frame : 0 },
            set: { newValue in
                guard draft.media.keyframes.indices.contains(index) else { return }
                draft.media.keyframes[index].frame = Self.snappedFrame(newValue, temporal: temporal, frames: clipFrames)
                draft.media.keyframes.sort { $0.frame < $1.frame }
            }
        )
    }

    private func attachmentBinding(index: Int) -> Binding<MediaWell.Attachment?> {
        Binding(
            get: {
                guard draft.media.keyframes.indices.contains(index) else { return nil }
                let keyframe = draft.media.keyframes[index]
                return MediaWell.Attachment(base64: keyframe.image, name: keyframe.name ?? "")
            },
            set: { newValue in
                guard draft.media.keyframes.indices.contains(index), let newValue else { return }
                draft.media.keyframes[index].image = newValue.base64
                draft.media.keyframes[index].name = newValue.name
            }
        )
    }

    private var ceiling: Int { Swift.max(clipFrames - 1, 0) }
    private var clipFrames: Int { draft.frames ?? temporal?.frames.default ?? 1 }
}

extension KeyframeTable {
    /// A requested frame, clamped strictly below `frames` -- never a value
    /// the server would refuse (`validation.rs:1705-1711`).
    static func snappedFrame(_ requested: Int, temporal: TemporalProfile, frames: Int) -> Int {
        let ceiling = Swift.max(frames - 1, 0)
        return Swift.min(Swift.max(temporal.snap(requested), 0), ceiling)
    }
}
