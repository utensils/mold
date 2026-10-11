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
    let recipe: GenerationRecipe
    @Binding var draft: RenderDraft
    @Environment(GenerateController.self) var controller
    private var temporal: TemporalProfile? { recipe.temporal }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(draft.media.keyframes.enumerated()), id: \.offset) { index, keyframe in
                row(index: index, keyframe: keyframe)
            }
            addButton
        }
    }

    @ViewBuilder private func row(index: Int, keyframe: KeyframeCondition) -> some View {
        if let temporal {
            let session = ReferenceImportSession(controller: controller, recipe: recipe, media: draft.media)
            let frames = clipFrames
            HStack(spacing: 6) {
                Stepper(value: frameBinding(index: index, temporal: temporal),
                        in: 0...ceiling, step: Swift.max(temporal.frames.step, 1)) {
                    Text(keyframe.frame.formatted())
                        .monospacedDigit()
                        .frame(minWidth: 30, alignment: .trailing)
                }
                .help("Choose which frame of the clip this picture should guide")
                PictureWell(
                    rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
                    picture: keyframe.image, label: "Picture at frame \(keyframe.frame)",
                    pick: { picked in
                        guard frames == clipFrames,
                              session.isCurrent(controller: controller, media: draft.media) else { return }
                        _ = Self.replace(picked, at: index, expected: keyframe, in: &draft.media)
                    })
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
            let session = ReferenceImportSession(controller: controller, recipe: recipe, media: draft.media)
            let frames = clipFrames
            PictureWell(
                rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
                placeholder: "plus", actionTitle: "Add keyframe", label: "Add keyframe picture",
                pick: { picked in
                    guard frames == clipFrames,
                          session.isCurrent(controller: controller, media: draft.media) else { return }
                    draft.media.addingKeyframe(KeyframeCondition(
                        frame: nextFrame(temporal: temporal), image: picked.encoded, name: picked.name))
                })
                .help("Choose a picture from any machine, a file, or the clipboard to guide a moment in the clip")
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

    private var ceiling: Int { Swift.max(clipFrames - 1, 0) }
    private var clipFrames: Int { draft.frames ?? temporal?.frames.default ?? 1 }
}

extension KeyframeTable {
    /// A slow replacement can never write into a different row after an edit/removal.
    @discardableResult
    static func replace(_ picked: ImportedPicture, at index: Int, expected: KeyframeCondition,
                        in media: inout DraftMedia) -> Bool {
        guard media.keyframes.indices.contains(index), media.keyframes[index] == expected else { return false }
        media.keyframes[index].image = picked.encoded
        media.keyframes[index].name = picked.name
        return true
    }

    /// A requested frame, clamped strictly below `frames` -- never a value
    /// the server would refuse (`validation.rs:1705-1711`).
    static func snappedFrame(_ requested: Int, temporal: TemporalProfile, frames: Int) -> Int {
        BoundaryFramePolicy.snapIndex(requested, temporal: temporal, frames: frames)
    }
}
