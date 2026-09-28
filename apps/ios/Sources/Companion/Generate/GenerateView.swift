import MoldClient
import SwiftUI

/// Generate (DESIGN.md §5.1): the canvas fills the screen; the composer --
/// wells, prompt, chips, estimate and Generate -- is a glass panel above the
/// tab bar that rides the keyboard. The toolbar holds the kind, the model
/// (plain name over its id in mono) and the machine.
struct GenerateView: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts
    @Environment(AppRouter.self) private var router
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
                GenerateCanvas()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .contentShape(.rect)
                    .onTapGesture { hideKeyboard() }
                    .safeAreaBar(edge: .bottom) {
                        Composer(showsOptions: $showsOptions, estimate: estimate)
                    }
            }
        }
        .navigationTitle(Destination.generate.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if !hosts.hosts.isEmpty {
                ToolbarItem(placement: .topBarLeading) { KindMenu() }
                ToolbarItem(placement: .principal) { ModelMenu() }
                ToolbarItem(placement: .topBarTrailing) { MachineMenu() }
            }
        }
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
        .task(id: hosts.upHosts.map(\.id)) { generate.settleChoice() }
        .task(id: estimateKey) { await refreshEstimate() }
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

/// Still picture / Short clip / 3-D object. A menu on iPhone; the iPad has
/// the room, so it shows the same three as a segmented control.
struct KindMenu: View {
    @Environment(GenerateController.self) private var generate
    @Environment(\.horizontalSizeClass) private var width

    var body: some View {
        let binding = Binding(get: { generate.kind }, set: { generate.setKind($0) })
        if width == .regular {
            Picker("Kind", selection: binding) {
                ForEach(PrintKind.allCases, id: \.self) { Text($0.makeTitle).tag($0) }
            }
            .pickerStyle(.segmented)
            .fixedSize()
        } else {
            Menu {
                Picker("Kind", selection: binding) {
                    ForEach(PrintKind.allCases, id: \.self) { kind in
                        Label(kind.makeTitle, systemImage: kind.makeSymbol).tag(kind)
                    }
                }
            } label: {
                Label(generate.kind.makeTitle, systemImage: generate.kind.makeSymbol)
            }
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
