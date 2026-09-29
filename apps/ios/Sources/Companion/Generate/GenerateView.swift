import MoldClient
import SwiftUI

/// Generate (DESIGN.md §5.1): the canvas and a bounded composer above the
/// tab bar. The prompt stays in one hierarchy as the keyboard appears;
/// the model button opens the kind, model, recipe and machine chooser.
struct GenerateView: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts
    @Environment(AppRouter.self) private var router
    @Environment(\.dynamicTypeSize) private var size
    @Environment(\.verticalSizeClass) private var verticalSizeClass
    @State private var showsOptions = false
    @State private var estimate: String?

    var body: some View {
        Group {
            if hosts.hosts.isEmpty {
                EmptyState(title: String(localized: "Add a machine to start generating"),
                           symbol: Destination.generate.symbol,
                           message: String(localized: "Mold makes pictures on a computer you own.")) {
                    Button("Add a Machine…") { router.addMachine() }.prominentAction()
                }
            } else {
                if UIDevice.current.userInterfaceIdiom == .phone && generate.run == .idle {
                    ScrollView {
                        Composer(showsOptions: $showsOptions, estimate: estimate,
                                 maximumHeight: .infinity, inline: true,
                                 inlineAction: verticalSizeClass == .compact)
                            // The pinned action overlays the scroll view on iOS;
                            // leave enough travel to lift Options above it.
                            .padding(.bottom, verticalSizeClass == .compact ? 0 : (size.isAccessibilitySize ? 160 : 112))
                    }
                    .scrollDismissesKeyboard(.interactively)
                    .accessibilityIdentifier("phone-generate-form")
                    .safeAreaInset(edge: .bottom, spacing: 0) {
                        if verticalSizeClass != .compact {
                            GenerateRow(estimate: estimate)
                                .padding(16)
                                .frame(maxWidth: .infinity)
                                .background(Color(uiColor: .systemBackground))
                        }
                    }
                } else {
                    GeometryReader { geometry in
                    Group {
                        if generate.run == .idle && size.isAccessibilitySize {
                            Color.clear
                        } else {
                            GenerateCanvas()
                        }
                    }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .contentShape(.rect)
                        .onTapGesture { hideKeyboard() }
                        .safeAreaBar(edge: .bottom) {
                            Composer(showsOptions: $showsOptions, estimate: estimate,
                                     maximumHeight: geometry.size.height * Self.composerHeightFraction(
                                        run: generate.run, accessibility: size.isAccessibilitySize))
                        }
                    }
                }
            }
        }
        .navigationTitle(Destination.generate.title)
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showsOptions) { MoreOptionsSheet() }
        .overlay(alignment: .top) {
            VStack(spacing: 8) {
                FailureBanner()
                if !hosts.hosts.isEmpty { FromShareCard() }
            }
        }
        .onChange(of: router.pendingReuse) { _, entry in
            guard let entry else { return }
            generate.reuse(entry)
            router.pendingReuse = nil
        }
        .onChange(of: hosts.models, initial: true) { _, _ in generate.settleChoice() }
        .onChange(of: hosts.upHosts.map(\.id)) { _, _ in generate.settleChoice() }
        .task(id: estimateKey) { await refreshEstimate() }
    }

    /// Idle accessibility text gets the canvas space it needs; submitting,
    /// progress, results and failures keep their visible canvas.
    static func composerHeightFraction(run: RunState, accessibility: Bool) -> CGFloat {
        accessibility && run == .idle ? 0.9 : 0.55
    }

    private var estimateKey: String {
        "\(generate.modelName ?? "")|\(generate.target?.id.uuidString ?? "")|\(generate.draft.width)x\(generate.draft.height)|\(generate.draft.steps)|\(generate.draft.batchSize)|\(generate.draft.frames ?? 0)"
    }

    /// Debounced; a guess is said as a guess ("about"), never as a measurement.
    private func refreshEstimate() async {
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
