import MoldClient
import MoldStyle
import SwiftUI

/// A short list of clips, sharing the ordinary model, canvas and sound controls.
struct NativeSequenceEditor: View {
    @Bindable var session: NativeSequenceSession
    @Environment(\.dynamicTypeSize) private var typeSize
    let fps: Int
    let frameRange: ClosedRange<Int>
    let frameStep: Int
    let maximumClips: Int
    var carryMode: String? = nil
    var baseSeed: UInt64? = nil

    var body: some View {
        if let draft = session.draft {
            VStack(alignment: .leading, spacing: 12) {
                if let title = session.continuationTitle, let host = session.continuationHost {
                    Text("Continuing \(title) on \(host.name)")
                        .font(.headline).fixedSize(horizontal: false, vertical: true)
                    Text("The next part saves as a separate video, using the original model, picture size and sound settings.")
                        .font(.caption).foregroundStyle(sequenceSecondary)
                }
                if let notice = session.settingsReuseNotice {
                    Text(notice).font(.callout).fixedSize(horizontal: false, vertical: true)
                }
                if session.needsReferenceDecision {
                    Text("Restore the original references, attach new ones, or continue without references.")
                        .font(.callout).fixedSize(horizontal: false, vertical: true)
                    Button("Restore original references") { session.restoreOriginalReferences() }
                    Button("Continue without references") { session.continueWithoutReferences() }
                }
                if carryMode == "independent" {
                    Text("These clips join as separate shots. Motion does not carry between them.")
                        .font(.caption).foregroundStyle(sequenceSecondary)
                }
                ForEach(Array(draft.stages.enumerated()), id: \.element.id) { index, stage in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Clip \(index + 1)").font(.headline)
                            Spacer(minLength: 8)
                            Menu {
                                Button("Move earlier") { session.draft?.move(from: index, to: index - 1) }
                                    .disabled(index == 0)
                                Button("Move later") { session.draft?.move(from: index, to: index + 1) }
                                    .disabled(index + 1 == draft.stages.count)
                                if session.settingsReuseNotice != nil {
                                    Button("Use global inputs and controls") { session.useGlobalSettings(for: stage.id) }
                                        .disabled(session.settingsInputsNeedRestoration)
                                }
                                Button("Remove clip", role: .destructive) { session.draft?.remove(id: stage.id) }
                                    .disabled(draft.stages.count <= 1)
                            } label: { Label("Clip \(index + 1) options", systemImage: "ellipsis.circle") }
                            .labelStyle(.iconOnly)
                            .accessibilityLabel("Clip \(index + 1) options")
                            .frame(minWidth: 44, minHeight: 44)
                        }
                        TextField("Describe clip \(index + 1)…", text: stagePrompt(index), axis: .vertical)
                            .lineLimit(2 ... 5)
                            .accessibilityLabel("Clip \(index + 1) prompt")
                            .accessibilityIdentifier("sequence-prompt-\(index)")
                        if session.settingsReuseNotice != nil {
                            ForEach(Self.savedControls(stage, baseSeed: baseSeed), id: \.self) { value in
                                Text(value).font(.caption).foregroundStyle(sequenceSecondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        Stepper(value: stageFrames(index), in: frameRange, step: max(1, frameStep)) {
                            Text("Length: \((Double(stage.frames) / Double(max(1, fps))).formatted(.number.precision(.fractionLength(1 ... 2)))) seconds")
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .accessibilityIdentifier("sequence-length-\(index)")
                    }
                    .padding(12)
                    .background(.quaternary, in: .rect(cornerRadius: 10))
                }
                let actions = typeSize.isAccessibilitySize ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8)) : AnyLayout(HStackLayout(spacing: 12))
                actions {
                    Button("Add another clip", systemImage: "plus") { session.draft?.append() }
                        .disabled(draft.stages.count >= maximumClips)
                        .accessibilityIdentifier("sequence-add-clip")
                    if !typeSize.isAccessibilitySize { Spacer(minLength: 8) }
                    Button("Start fresh") { session.startFresh() }
                        .accessibilityIdentifier("sequence-start-fresh")
                }
                .frame(minHeight: 44)
                Text(draft.stages.count == 1 ? "1 clip · one video per part" : "\(draft.stages.count) clips · one video per part")
                    .font(.caption).foregroundStyle(sequenceSecondary)
                if let plan = session.validatedPlan {
                    Text("Joined length: \(Double(plan.estimatedDurationMs) / 1000, format: .number.precision(.fractionLength(1 ... 2))) seconds")
                        .font(.callout).accessibilityIdentifier("sequence-planned-duration")
                    ForEach(plan.warnings, id: \.self) { warning in
                        Text(warning).font(.caption).foregroundStyle(sequenceSecondary)
                    }
                }
                if let message = session.message {
                    Text(message).font(.callout).fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("sequence-status")
                }
            }
            .disabled(session.preparing)
        }
    }
    static func savedControls(_ stage: SequenceDraft.Stage, baseSeed: UInt64?) -> [String] {
        var lines: [String] = []
        if stage.input.sourceImage != nil { lines.append("Original first frame") }
        for reference in stage.generationReferences ?? [] {
            lines.append("Original \(reference.kind) reference: \(reference.provenance?.name ?? "Reference")")
        }
        for frame in stage.keyframes ?? [] { lines.append("Original frame \(frame.frame): \(frame.name ?? "Image")") }
        if let seed = baseSeed, let offset = stage.input.seedOffset { lines.append("Seed: \(seed ^ offset)") }
        if let negative = stage.input.negativePrompt, !negative.isEmpty { lines.append("Negative prompt: \(negative)") }
        for adapter in stage.input.loras ?? [] {
            let name = adapter.name ?? adapter.path.split(separator: "/").last.map(String.init) ?? "Adapter"
            lines.append("Adapter: \(name) · \(adapter.scale)")
        }
        if stage.transition != "smooth" { lines.append("Join: \(stage.transition.capitalized)") }
        return lines
    }

    private func stagePrompt(_ index: Int) -> Binding<String> {
        Binding(get: { session.draft?.stages[safe: index]?.prompt ?? "" }, set: {
            guard session.draft?.stages.indices.contains(index) == true else { return }
            session.draft?.stages[index].prompt = $0
        })
    }
    private func stageFrames(_ index: Int) -> Binding<Int> {
        Binding(get: { session.draft?.stages[safe: index]?.frames ?? frameRange.lowerBound }, set: {
            guard session.draft?.stages.indices.contains(index) == true else { return }
            session.draft?.stages[index].frames = $0
        })
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? { indices.contains(index) ? self[index] : nil }
}

struct NativeSequenceJobs: View {
    @Bindable var session: NativeSequenceSession
    let backend: (NativeSequenceSession.Job) -> (any MoldChainBackend)?
    var open: ((MoldHost, String) -> Void)?
    var restart: ((NativeSequenceSession.Job) -> Void)?
    var discard: ((NativeSequenceSession.Job) -> Void)?
    var canRestart: (NativeSequenceSession.Job) -> Bool = { _ in false }
    var body: some View {
        if let notice = session.historyNotice { Text(notice).fixedSize(horizontal: false, vertical: true) }
        ForEach(session.jobs) { job in
            VStack(alignment: .leading, spacing: 8) {
                Label("Video · \(job.host.name)", systemImage: "film.stack")
                    .font(.headline)
                if let script = job.detail?.script, let prompt = script.stage.first?.prompt {
                    Text(prompt).font(.subheadline).lineLimit(2)
                    Text(script.chain.model).font(.caption).foregroundStyle(sequenceSecondary)
                }
                Text(job.message).fixedSize(horizontal: false, vertical: true)
                if let detail = job.detail {
                    if let filename = detail.galleryFilename {
                        if let open {
                            Button("Show video in Library") { open(job.host, filename) }
                        } else { Text(filename).font(.caption).textSelection(.enabled) }
                    }
                    if detail.state == .completed || detail.state == .cancelled {
                        Button("Dismiss completed video") { session.dismiss(job) }.frame(minHeight: 44)
                    }
                    if let backend = backend(job) {
                        if ChainJobRecoveryPolicy.canResume(state:detail.state,ephemeral:detail.ephemeral == true) {
                            Button("Resume") { session.resume(job,backend:backend) }.frame(minHeight:44)
                        }
                        if ChainJobRecoveryPolicy.canCancel(state:detail.state) {
                            Button("Stop",role:.destructive) { session.cancel(job,backend:backend) }
                                .disabled(job.cancelling).frame(minHeight:44)
                        }
                        if canRestart(job) {
                            if let restart, detail.state == .failed || detail.state == .interrupted || detail.state == .paused {
                                Button("Start again…") {restart(job)}.frame(minHeight:44)
                            }
                            if let discard, detail.state == .failed || detail.state == .interrupted {
                                Button("Delete job…",role:.destructive) {discard(job)}.frame(minHeight:44)
                            }
                        }
                    }
                } else if let backend = backend(job) {
                    Button("Stop", role: .destructive) { session.cancel(job, backend: backend) }
                        .disabled(job.cancelling).frame(minHeight: 44)
                }
            }
            .padding(.vertical, 8)
        }
    }
}

private var sequenceSecondary: Color {
#if os(iOS)
    .secondaryText
#else
    .secondary
#endif
}
