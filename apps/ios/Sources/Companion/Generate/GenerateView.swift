import MoldClient
import SwiftUI

/// Generate: a stable composer and pinned submission action above the
/// tab bar. The prompt stays in one hierarchy as the keyboard appears;
/// the model button opens the kind, model, recipe and machine chooser.
struct GenerateView: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts
    @Environment(AppRouter.self) private var router
    @Environment(\.dynamicTypeSize) private var size
    @State private var showsOptions = false
    @State private var showsPromptEditor = false
    @State private var expansionUndo = PromptExpansionUndo()
    @State private var expandingPrompt = false
    @State private var promptSuggestions: [String] = []
    @State private var estimate: String?
    @State private var choosingSourceRole = false

    var body: some View {
        Group {
            if hosts.hosts.isEmpty {
                EmptyState(title: String(localized: "Add a machine to start generating"),
                           symbol: Destination.generate.symbol,
                           message: String(localized: "Mold makes pictures on a computer you own.")) {
                    Button("Add a Machine…") { router.addMachine() }.prominentAction()
                }
            } else {
                ScrollView {
                    Composer(showsOptions: $showsOptions, showsPromptEditor: $showsPromptEditor, expansionUndo: $expansionUndo, expandingPrompt: $expandingPrompt,
                             promptSuggestions: $promptSuggestions, estimate: estimate,
                             maximumHeight: .infinity, inline: true)
                        .padding(.bottom, 24)
                }
                .scrollDismissesKeyboard(.interactively)
                .accessibilityIdentifier("phone-generate-form")
                .safeAreaInset(edge: .bottom, spacing: 0) {
                    VStack(spacing: 8) {
                        if generate.run == .submitting, let preparing = generate.referencePreparation { Text(preparing).font(.subheadline).accessibilityIdentifier("reference-preparation-status") }
                        QueueStatusLink()
                        GenerateRow(estimate: estimate, promptEditorPresented: showsPromptEditor)
                    }
                    .padding(16)
                    .frame(maxWidth: .infinity)
                    .background(Color(uiColor: .systemBackground))
                }

            }
        }
        .onChange(of: generate.draft) { _, draft in expansionUndo.observe(draft) }
        .onChange(of: generate.sequence.draft) { _, draft in
            if draft != nil { generate.saveDraft() }
        }
        .navigationTitle(Destination.generate.title)
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showsOptions) { MoreOptionsSheet() }
        .safeAreaInset(edge: .top, spacing: 0) {
            VStack(spacing: 8) {
                FailureBanner()
                if !hosts.hosts.isEmpty { FromShareCard() }
            }
        }
        .onChange(of: router.pendingReuse, initial: true) { _, entry in
            guard let entry else { return }
            generate.reuse(entry)
            router.pendingReuse = nil
        }
        .confirmationDialog("Use Picture as Camera View", isPresented: $choosingSourceRole, titleVisibility: .visible) {
            ForEach(generate.recipe?.capabilities.mesh?.namedViews?.roles ?? [], id: \.self) { role in
                Button(role.rawValue.capitalized) {
                    guard let entry = router.pendingSource else { return }
                    router.pendingSource = nil
                    Task { await generate.useAsSource(entry, role: role) }
                }
            }
            Button("Cancel", role: .cancel) { router.pendingSource = nil }
        }
        .task(id: router.pendingSourceRequest) {
            guard let entry = router.pendingSource else { return }
            if router.pendingSourceAttachment == nil, generate.recipe?.capabilities.mesh?.namedViews?.mode.isVisible == true {
                choosingSourceRole = true
                return
            }
            await generate.useAsSource(entry, attachment: router.pendingSourceAttachment)
            if router.pendingSource?.id == entry.id { router.pendingSource = nil }
        }
        .onChange(of: hosts.models, initial: true) { _, _ in generate.settleChoice() }
        .onChange(of: hosts.upHosts.map(\.id)) { _, _ in generate.settleChoice() }
        .task(id: DurationPlanKey(host: generate.target,
            instance: generate.target.flatMap { hosts.instanceID(of: $0.id) },
            enabled: generate.canAuthorVideoDuration, authoring: generate.videoDuration,
            draft: generate.draft, model: generate.modelName)) {
            generate.refreshDurationPlan()
        }
        .onChange(of: generate.videoDuration) { _, _ in generate.saveDraft() }
        .task(id: estimateKey) { await refreshEstimate() }
        .task(id: sequenceCapabilityKey) { await generate.refreshSequenceLimits() }
        .task(id: durationSourceKey) { await generate.restoreDurationSource() }
        .onChange(of: router.pendingVideoExtension, initial: true) { _, entry in
            guard let entry else { return }
            router.pendingVideoExtension = nil
            generate.extendVideo(entry)
        }
        .onChange(of: router.pendingContinuation, initial: true) { _, entry in
            guard let entry else { return }
            router.pendingContinuation = nil
            generate.continueVideo(entry)
        }
    }

    private struct DurationPlanKey: Equatable {
        let host: MoldHost?
        let instance: String?
        let enabled: Bool
        let authoring: VideoDurationAuthoring?
        let draft: RenderDraft
        let model: String?
    }

    private struct SequenceCapabilityKey: Equatable {
        let host: MoldHost?
        let instance: String?
        let model: String?
        let fps: Int?
        let revision: String?
        let restoringSource: Bool
    }
    private var sequenceCapabilityKey: SequenceCapabilityKey {
        SequenceCapabilityKey(host: generate.target,
            instance: generate.target.flatMap { hosts.instanceID(of: $0.id) },
            model: generate.modelName, fps: generate.draft.fps,
            revision: generate.target.flatMap { host in
                if case let .up(status) = hosts.reachability(of: host) { status.gitSha } else { nil }
            }, restoringSource: generate.sequence.restoringSource)
    }

    private var estimateKey: String {
        "\(generate.modelName ?? "")|\(generate.target?.id.uuidString ?? "")|\(generate.draft.width)x\(generate.draft.height)|\(generate.draft.steps)|\(generate.draft.batchSize)|\(generate.draft.frames ?? 0)|\(generate.draft.media.generationReferences.map { $0.provenance?.sha256 ?? "\($0.kind):\($0.durationMs ?? 0):\($0.width ?? 0)x\($0.height ?? 0)" }.joined(separator: ","))"
    }

    /// Debounced; a guess is said as a guess ("about"), never as a measurement.
    private func refreshEstimate() async {
        guard !generate.usesVideoDuration else { estimate = nil; return }
        guard !generate.draft.media.generationReferences.contains(where: { $0.freshVideo != nil }) else { estimate = nil; return }
        try? await Task.sleep(for: .milliseconds(400))
        guard !Task.isCancelled, let host = generate.target, let model = generate.modelName else {
            estimate = nil
            return
        }
        let request = RenderRequest.placement(generate.draft, model: model,
                                              maxIdentityPhotos: hosts.capabilities[host.id]?.maxIdentityPhotos ?? 0)
        guard let answer = try? await hosts.backend(for: host).placementPreview(request, copies: generate.draft.batchSize),
              !Task.isCancelled else { return }
        estimate = answer.candidate?.predictedDuration.map {
            String(localized: "about \($0.formatted(.units(allowed: [.minutes, .seconds], width: .narrow)))")
        }
    }
}

/// A stable menu at every text size: changing to a segmented picker during
/// Dynamic Type changes replaces the selected label's accessibility node.
struct KindMenu: View {
    @Environment(GenerateController.self) private var generate

    var body: some View {
        let binding = Binding(get: { generate.kind }, set: { generate.setKind($0) })
        Menu {
            Picker("Kind", selection: binding) {
                ForEach(PrintKind.allCases, id: \.self) { kind in
                    Label(kind.makeTitle, systemImage: kind.makeSymbol).tag(kind)
                }
            }
        } label: {
            Label(generate.kind.makeTitle, systemImage: generate.kind.makeSymbol)
                .fixedSize(horizontal: false, vertical: true)
                .foregroundStyle(.primary)
                .frame(minHeight: 44)
                .contentShape(Rectangle())
        }
    }
}

extension PrintKind {
    var makeTitle: String {
        switch self {
        case .picture: String(localized: "Still picture")
        case .clip: String(localized: "Short clip")
        case .mesh: String(localized: "3-D object")
        }
    }

    var makeSymbol: String {
        switch self {
        case .picture: "photo"
        case .clip: "film"
        case .mesh: "cube"
        }
    }
}

extension View {
    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

extension GenerateView {
    private struct DurationSourceKey: Hashable {
        let source: VideoJobSource?
        let sourceOwner: VideoLibrarySourceOwner?
        let host: MoldHost?
        let instance: String?
        let reachable: Bool
    }
    private var durationSourceKey: DurationSourceKey {
        let host = generate.videoDuration?.sourceOwner.flatMap { hosts.host($0.hostId) }
        return .init(source: generate.videoDuration?.draft.source, sourceOwner: generate.videoDuration?.sourceOwner,
                     host: host, instance: host.flatMap { hosts.instanceID(of: $0.id) },
                     reachable: host.map(hosts.isUp) ?? false)
    }
}
