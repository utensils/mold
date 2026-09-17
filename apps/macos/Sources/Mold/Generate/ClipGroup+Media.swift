import MoldClient
import MoldStyle
import SwiftUI
import UniformTypeIdentifiers

/// Keyframes, the extend continuation, and the two conditioning wells --
/// split from `ClipGroup.swift` purely for size.
extension ClipGroup {
    @ViewBuilder var keyframesSection: some View {
        if recipe?.capabilities.acceptsKeyframes == true {
            LabeledSection("Keyframes") {
                KeyframeTable(temporal: recipe?.temporal, draft: $draft)
            }
        }
    }

    @ViewBuilder var extendSection: some View {
        if recipe?.capabilities.supportsExtend == true {
            LabeledSection("Start from") {
                ExtendRow(temporal: recipe?.temporal, draft: $draft)
            }
        }
    }

    @ViewBuilder var audioFileSection: some View {
        if recipe?.capabilities.acceptsSourceAudio == true {
            LabeledSection("Voice / music") {
                MediaWell(
                    systemImage: "waveform", allowedTypes: [.audio],
                    placeholder: "No audio", attachment: audioFileBinding
                )
            }
        }
    }

    @ViewBuilder var sourceVideoSection: some View {
        if recipe?.capabilities.acceptsSourceVideo == true {
            LabeledSection("Source video") {
                MediaWell(
                    systemImage: "video", allowedTypes: [.movie],
                    placeholder: "No clip", attachment: sourceVideoBinding
                )
            }
        }
    }

    private var audioFileBinding: Binding<MediaWell.Attachment?> {
        Binding(
            get: {
                draft.media.audioFile.map { MediaWell.Attachment(base64: $0, name: draft.media.audioFileName ?? "") }
            },
            set: { newValue in
                draft.media.audioFile = newValue?.base64
                draft.media.audioFileName = newValue?.name
            }
        )
    }

    private var sourceVideoBinding: Binding<MediaWell.Attachment?> {
        Binding(
            get: {
                draft.media.sourceVideo.map {
                    MediaWell.Attachment(base64: $0, name: draft.media.sourceVideoName ?? "")
                }
            },
            set: { newValue in
                draft.media.sourceVideo = newValue?.base64
                draft.media.sourceVideoName = newValue?.name
            }
        )
    }
}
