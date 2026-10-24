import MoldClient
import SwiftUI

/// The composer (DESIGN.md §5.1): prompt, model, picture wells, the chip
/// row, then the estimate and Generate. At
/// accessibility sizes the chips fold into one Options button and Generate
/// takes the full width under the estimate. The window supplies a bounded
/// viewport, enlarged for idle accessibility text.
struct Composer: View {
    @Environment(GenerateController.self) private var generate
    @Environment(\.dynamicTypeSize) private var size
    @Binding var showsOptions: Bool
    @Binding var showsPromptEditor: Bool
    @Binding var expansionUndo: PromptExpansionUndo
    @Binding var expandingPrompt: Bool
    @Binding var promptSuggestions: [String]
    let estimate: String?
    let maximumHeight: CGFloat
    var inline = false
    var inlineAction = false
    @FocusState private var editing: Bool
    @Environment(HostStore.self) private var hosts
    @State private var historyHost: MoldHost?
    @AccessibilityFocusState private var editorOpenerFocused: Bool

    var body: some View {
        Group {
            if inline {
                content.padding(16)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        content.padding(14)
                    }
                    .scrollBounceBehavior(.basedOnSize)
                    .scrollEdgeEffectHidden(true)
                    .scrollDismissesKeyboard(.interactively)
                    .onChange(of: editing) { _, focused in
                        if focused { proxy.scrollTo("prompt", anchor: .top) }
                    }
                }
                .frame(maxHeight: maximumHeight)
                .clipped()
                .background(Color(uiColor: .systemBackground), in: .rect(cornerRadius: 16))
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("bottom-chrome")
                .padding(.horizontal, 12)
                .padding(.bottom, 6)
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                if !showsPromptEditor {
                    Spacer()
                    Button("Done") {
                        editing = false
                        // Clip rows own their text input outside this FocusState.
                        // End the active responder as well as ordinary prompt focus.
                        hideKeyboard()
                    }
                    .accessibilityIdentifier("keyboard-done")
                }
            }
        }
        .sheet(isPresented: $showsPromptEditor, onDismiss: { generate.saveDraft(); editorOpenerFocused = true }) {
            PromptEditorSheet(expansionUndo: $expansionUndo, expandingPrompt: $expandingPrompt,
                              promptSuggestions: $promptSuggestions)
        }
        .sheet(item: $historyHost) { host in
            PromptHistorySheet(host: host, hosts: hosts) { text in
                expansionUndo = PromptExpansionUndo()
                promptSuggestions = []
                PromptHistoryStore.recall(text, into: &generate.draft)
            }
        }
        .sheet(isPresented: Binding(get: { !showsPromptEditor && historyHost == nil && !promptSuggestions.isEmpty }, set: {
            if !$0 && !showsPromptEditor && historyHost == nil { promptSuggestions = [] }
        })) {
            SuggestionsSheet(suggestions: promptSuggestions) { chosen in
                expansionUndo.record(original: generate.draft, expanded: chosen)
                generate.draft.prompt = chosen
                promptSuggestions = []
            }
        }
    }

    private var content: some View {
        @Bindable var generate = generate
        return VStack(alignment: .leading, spacing: 12) {
            if inline {
                if size >= .xxLarge {
                    VStack(alignment: .leading, spacing: 12) {
                        KindMenu().disabled(generate.sequence.continuation != nil || generate.sequence.preparing)
                        MachineMenu().disabled(generate.sequence.continuation != nil || generate.sequence.preparing)
                    }
                } else {
                    HStack {
                        KindMenu().disabled(generate.sequence.continuation != nil || generate.sequence.preparing)
                        Spacer(minLength: 8)
                        MachineMenu().disabled(generate.sequence.continuation != nil || generate.sequence.preparing)
                    }
                }
            }
            if generate.sequence.draft != nil {
                let temporal = generate.recipe?.temporal
                let retainedFrames = generate.sequence.draft?.stages.first?.frames ?? 1
                NativeSequenceEditor(session: generate.sequence, fps: generate.draft.fps ?? 24,
                    frameRange: (generate.sequenceLimits?.authoring?.minFrames ?? temporal?.frames.min ?? retainedFrames) ... max(generate.sequenceLimits?.authoring?.minFrames ?? temporal?.frames.min ?? retainedFrames, min(temporal?.frames.max ?? retainedFrames, generate.sequenceLimits?.framesPerClipCap ?? temporal?.frames.max ?? retainedFrames)),
                    frameStep: generate.sequenceLimits?.authoring?.frameStep ?? temporal?.frames.step ?? 1, maximumClips: generate.sequenceLimits?.maxStages ?? 16,
                    carryMode: generate.sequenceLimits?.authoring?.carryMode, baseSeed: generate.draft.seed)
            } else if generate.recipe?.capabilities.promptRequirement != .ignored && !(generate.usesVideoDuration && generate.videoDuration?.draft.scenesEnabled == true && generate.videoDuration?.draft.scenes.isEmpty == false) {
                HStack(alignment: .top, spacing: 8) {
                    TextField("Prompt", text: $generate.draft.prompt,
                              prompt: Text(generate.kind == .clip ? "Describe a clip…" : "Describe a picture…")
                                .foregroundStyle(.secondaryText), axis: .vertical)
                        // At the smallest text one line is too short a
                        // target to hit; two reserved lines are not.
                        .lineLimit((size <= .small ? 2 : 1) ... 3)
                        .focused($editing)
                        .accessibilityIdentifier("generation-prompt")
                        .id("prompt")
                        .frame(minHeight: 44)
                    ExpandButton(undo: $expansionUndo, working: $expandingPrompt,
                                 suggestions: $promptSuggestions, shortcutEnabled: !showsPromptEditor)
                }
                Button("Edit prompt", systemImage: "square.and.pencil") {
                    editing = false
                    showsPromptEditor = true
                }
                .frame(minHeight: 44)
                .accessibilityIdentifier("prompt-editor-open")
                .accessibilityFocused($editorOpenerFocused)
            } else if generate.recipe?.capabilities.promptRequirement == .ignored {
                Text("This model works from a picture, not a description.")
                    .foregroundStyle(.secondaryText)
            }
            if generate.sequence.draft == nil {
            Button {
                historyHost = generate.target ?? hosts.preferredHost
            } label: {
                Label("Prompt History", systemImage: "clock.arrow.circlepath")
                    .frame(minWidth: 44, minHeight: 44)
                    .contentShape(Rectangle())
            }
                .accessibilityIdentifier("prompt-history")
                .disabled(generate.target == nil && hosts.preferredHost == nil)
            }
            if let error = generate.durationSourceError,
               generate.videoDuration?.draft.source == nil || !generate.usesVideoDuration {
                NativeVideoSourceFailure(message: error, keepDraft: generate.keepDraftAfterSourceFailure)
            }
            if generate.kind == .clip, generate.sequence.draft == nil, let temporal = generate.recipe?.temporal {
                ComposerLength(temporal: temporal)
            }
            if generate.usesVideoDuration, let authoring = generate.videoDuration {
                NativeDurationEditor(draft: Binding(get: { generate.videoDuration?.draft ?? authoring.draft }, set: {
                    generate.videoDuration?.draft = $0
                }), planner: generate.videoPlanner, prompt: generate.draft.prompt,
                    grid: generate.recipe?.temporal.map { VideoLengthGrid.duration(temporal: $0, fps: generate.draft.fps,
                        limits: generate.sequenceLimits, plan: generate.videoPlanner.plan) },
                    reuseNotice: generate.videoDuration?.reuseNotice, sourceTitle: generate.videoDuration?.sourceOwner?.title,
                    sourceMachine: generate.videoDuration?.sourceOwner.flatMap { hosts.host($0.hostId)?.name },
                    busy: generate.durationTask != nil,
                    originalReferencesReplaced: !generate.draft.media.generationReferences.isEmpty,
                    originalReferences: generate.durationSource?.originalReferences,
                    inheritedRecipeNotes: generate.durationSource?.inheritedRecipeNotes ?? [],
                    sourceNotice: generate.durationSourceError,
                    retrySource: { Task { await generate.restoreDurationSource() } }, retry: { generate.refreshDurationPlan() },
                    startFresh: { generate.videoDuration?.startFresh(); generate.durationSource = nil; generate.durationSourceError = nil; generate.saveDraft() })
                if let error = generate.durationError ?? generate.videoJobs.error, error != generate.durationSourceError {
                    Text(error).font(.callout).fixedSize(horizontal: false, vertical: true)
                }
            }
            if generate.kind == .clip, !generate.usesVideoDuration, generate.sequence.draft == nil, generate.sequenceLimits?.supportsSequence == true {
                Button("Add another clip", systemImage: "plus") {
                    generate.sequence.begin(prompt: generate.draft.prompt,
                        frames: generate.draft.frames ?? generate.sequenceLimits!.framesPerClipRecommended)
                }.frame(minHeight: 44).accessibilityIdentifier("sequence-begin")
            }
            ModelMenu().disabled(generate.sequence.continuation != nil || generate.sequence.preparing)
            if let notice = generate.sequence.message, generate.sequence.draft == nil {
                Text(notice).font(.callout).fixedSize(horizontal: false, vertical: true)
            }
            if generate.sequence.draft != nil {
                Text("Reference media applies to every clip. The opening image starts the video; the closing image ends it.")
                    .font(.caption).foregroundStyle(.secondaryText)
            }
            PictureWells().disabled(generate.sequence.preparing || generate.sequence.continuation != nil)
            TypedReferenceWells().disabled(generate.sequence.preparing)
            BoundaryFrameWells().disabled(generate.sequence.preparing)
            if let notice = generate.retainedReuse.notice {
                Label(notice, systemImage: "photo.on.rectangle")
                    .font(.subheadline)
                    .foregroundStyle(.secondaryText)
                if generate.retainedReuse.canRetry {
                    Button("Retry retained media", systemImage: "arrow.clockwise") {
                        generate.retainedReuse.retry(controller: generate)
                    }
                }
                if generate.retainedReuse.canDiscard {
                    Button("Remove retained sources", systemImage: "xmark.circle") {
                        generate.retainedReuse.clear()
                    }
                }
            }
            if let blocker = generate.blocker, !generate.run.isBusy {
                Label(blocker, systemImage: "exclamationmark.circle")
                    .font(.subheadline)
                    .foregroundStyle(.secondaryText)
            }
            if inline || size >= .xxLarge {
                optionsButton
            } else {
                ViewThatFits(in: .horizontal) {
                    ChipRow(style: .full, showsOptions: $showsOptions)
                    ChipRow(style: .short, showsOptions: $showsOptions)
                    optionsButton
                }
            }
            if !inline || inlineAction { GenerateRow(estimate: estimate, promptEditorPresented: showsPromptEditor) }
            if case let .failed(reason) = generate.run {
                Text(reason).foregroundStyle(.secondaryText)
            }
        }
    }

    private var optionsButton: some View {
        Button { showsOptions = true } label: {
            Label("Options", systemImage: "slider.horizontal.3").frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
    }
}

/// The estimate and Generate: side by side, or stacked with Generate full
/// width at accessibility sizes. Generate is never Stop.
struct GenerateRow: View {
    @Environment(GenerateController.self) private var generate
    @Environment(\.dynamicTypeSize) private var size
    let estimate: String?
    var promptEditorPresented = false

    static func stacks(at size: DynamicTypeSize, phone: Bool = false) -> Bool { phone || size >= .xxLarge }

    var body: some View {
        let stacked = Self.stacks(at: size, phone: UIDevice.current.userInterfaceIdiom == .phone)
        let layout = stacked ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
                             : AnyLayout(HStackLayout(spacing: 12))
        layout {
            Text(generate.sequence.draft == nil && !generate.usesVideoDuration ? (estimate ?? " ") : " ").monospacedDigit().foregroundStyle(.secondaryText)
                .accessibilityHidden(estimate == nil || generate.sequence.draft != nil || generate.usesVideoDuration)
            if !stacked { Spacer(minLength: 0) }
            Button { if !promptEditorPresented { generate.generate() } } label: {
                Label(generate.sequence.draft == nil && !generate.usesVideoDuration ? "Generate" : "Generate video", systemImage: "wand.and.sparkles")
                    .frame(maxWidth: stacked ? .infinity : nil)
            }
            .prominentAction()
            .controlSize(.large)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(generate.blocker != nil || promptEditorPresented || generate.sequence.preparing || generate.durationTask != nil)
            .accessibilityIdentifier("submit-generation")
            .accessibilityShowsLargeContentViewer()
            .sensoryFeedback(.impact(weight: .light), trigger: generate.queued.count + (generate.run.isBusy ? 1 : 0))
        }
    }
}

/// Shape · Steps · Batch · Length · More options, in two widths.
struct ChipRow: View {
    enum Style { case full, short }
    @Environment(GenerateController.self) private var generate
    let style: Style
    @Binding var showsOptions: Bool

    var body: some View {
        HStack(spacing: 8) {
            if let recipe = generate.recipe {
                ShapeChip(resolution: recipe.resolution, short: style == .short)
                    .disabled(generate.sequence.continuation != nil)
                if recipe.steps.mode != .fixed {
                    StepperChip(title: String(localized: "Steps"), value: generate.draft.steps,
                                range: recipe.steps.min ... recipe.steps.max) { generate.draft.steps = $0 }
                }
                if generate.kind == .picture {
                    StepperChip(title: String(localized: "Batch"), value: generate.draft.batchSize,
                                range: 1 ... max(1, generate.target.flatMap { generate.hosts.capabilities[$0.id]?.maxBatchOutputs } ?? 4)) {
                        generate.draft.batchSize = $0
                    }
                }
            }
            Button { showsOptions = true } label: {
                Label("More Options", systemImage: "slider.horizontal.3")
                    .labelStyle(.iconOnly)
                    .frame(minWidth: 44, minHeight: 36)
            }
            .buttonStyle(.bordered)
            .accessibilityShowsLargeContentViewer()
        }
    }
}
