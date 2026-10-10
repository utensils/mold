import MoldClient
import MoldStyle
import SwiftUI
import AppKit

/// The floating panel: what to make, how to make it, and the button.
struct PromptPanel: View {
    /// The capsule's ceiling. `PromptLip` matches it so the lip sits flush
    /// under the capsule it came from.
    static let maxWidth: CGFloat = 760

    let recipe: GenerationRecipe?
    @Binding var draft: RenderDraft
    let model: Model?
    let host: MoldHost?
    @Binding var destination: Destination
    let submit: () -> Void
    let cancel: () -> Void
    /// Stops every batch this pane has admitted, not just the one on screen
    /// (M8 decision 8). Neither this nor `cancel` takes a backend any more --
    /// each batch resolves its own machine.
    let stopAll: () -> Void
    let maxBatch: Int
    /// This machine's own chain limits for the chosen model.
    let chainLimits: ChainLimits?
    let maxHeight: CGFloat

    /// Not `private`: `PromptPanel+Actions`, an extension in another file,
    /// reads the run and the queue depth to build the trailing button group.
    @Environment(ReuseStore.self) var reuse
    @Environment(DraftPersistence.self) var drafts
    @Environment(GenerateController.self) var controller
    @Environment(ExpandStore.self) private var expansions
    @Environment(PromptHistoryStore.self) private var history
    private enum FocusedField: Hashable { case prompt, negative, editPrompt }
    @FocusState private var focusedField: FocusedField?
    @State var isEditingPrompt = false
    @State private var cycler = PromptHistoryCycler()
    @State private var keyMonitor: Any?
    @State private var contentHeight: CGFloat = 0
    @State private var actionsHeight: CGFloat = 0
    @State private var stepsHeight: CGFloat = 0
    @AppStorage("generatePromptEditorHeight", store: AppStorageSuite.defaults)
    private var preferredPromptHeight = Double(PromptEditorHeight.initial)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let steps = controller.run.steps {
                StepSegments(done: steps.done, total: steps.total)
                    .onGeometryChange(for: CGFloat.self) { $0.size.height } action: {
                        stepsHeight = $0
                    }
            }
            if let recipe {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        prompt(recipe)
                        promptTools(recipe)
                        Divider()
                        ControlsRow(recipe: recipe, model: model, maxBatch: maxBatch,
                                    chainLimits: chainLimits, draft: $draft)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .onGeometryChange(for: CGFloat.self) { $0.size.height } action: {
                        contentHeight = $0
                    }
                }
                .frame(height: min(contentHeight, scrollHeight))
                actions(recipe)
                    .onGeometryChange(for: CGFloat.self) { $0.size.height } action: {
                        actionsHeight = $0
                    }
            } else {
                Text("Pick a model to see its controls.")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .panel(.floating)
        .frame(maxWidth: Self.maxWidth)
        .frame(maxHeight: maxHeight, alignment: .bottom)
        // Published the same way the Library's title and tag fields already
        // do, so `ResultStrip`'s arrow-key shortcuts stand down for a caret
        // here exactly as they do for one there.
        .focusedValue(\.editingText, focusedField == .prompt || focusedField == .negative ? true : nil)
        .onAppear { installHistoryKeyMonitor() }
        .onDisappear { removeHistoryKeyMonitor() }
        .sheet(isPresented: $isEditingPrompt, onDismiss: { focusedField = .editPrompt }) {
            if let recipe {
                PromptEditorSheet(draft: $draft, recipe: recipe, host: host, destination: $destination)
            }
        }
        .task(id: host?.id) {
            guard let host else { return }
            await history.refresh(on: host.id)
        }
        .onChange(of: host?.id, initial: true) { _, _ in loadHistory() }
        .onChange(of: host.flatMap { history.byHost[$0.id] }) { _, _ in loadHistory() }
    }

    @ViewBuilder private func prompt(_ recipe: GenerationRecipe) -> some View {
        switch recipe.capabilities.promptRequirement {
        case .ignored:
            // No text encoder anywhere in this family. A prompt box here would
            // be furniture -- the recipe's own words say why.
            Text(recipe.capabilities.prompt?.reason ?? "This model doesn't read a prompt.")
                .font(.callout)
                .foregroundStyle(.secondary)
        default:
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    PromptResizeHandle(preferredHeight: $preferredPromptHeight, available: min(144, promptAvailableHeight))
                    TextEditor(text: $draft.prompt)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .frame(height: PromptEditingLayout.compactHeight(preferred: preferredPromptHeight, available: promptAvailableHeight))
                        .overlay(alignment: .topLeading) {
                            if draft.prompt.isEmpty {
                                Text(placeholder(recipe)).foregroundStyle(.secondary)
                                    .padding(.leading, 5).padding(.top, 1)
                                    .allowsHitTesting(false)
                                    .accessibilityHidden(true)
                            }
                        }
                        .accessibilityLabel(placeholder(recipe))
                        .accessibilityIdentifier("generate-prompt-editor")
                        .focused($focusedField, equals: .prompt)
                    if recipe.capabilities.negativePrompt?.isAvailable == true {
                        TextField("Avoid…", text: $draft.negativePrompt, axis: .vertical)
                            .textFieldStyle(.plain)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .lineLimit(1...3)
                            // Shares the one focus state with the prompt
                            // field above: a bare `Bool` binding answers
                            // "is either of these two typing", which is
                            // exactly what a caret's claim on an arrow key
                            // needs.
                            .focused($focusedField, equals: .negative)
                    }
                }
                ImageConditioningWells(recipe: recipe, model: model, draft: $draft)
            }
        }
    }

    /// The wand's split button and, once a rewrite has been accepted, the
    /// way back out of it -- under the prompt rather than a glyph pinned to
    /// its corner (M8 decision 4).
    @ViewBuilder private func promptTools(_ recipe: GenerationRecipe) -> some View {
        if recipe.capabilities.promptRequirement != .ignored {
            HStack(spacing: 10) {
                Button("Edit prompt", systemImage: "square.and.pencil") { isEditingPrompt = true }
                    .accessibilityIdentifier("edit-prompt")
                    .focused($focusedField, equals: .editPrompt)
                if let host {
                    PromptWand(recipe: recipe, host: host, draft: $draft, destination: $destination,
                               presentationEnabled: !isEditingPrompt)
                }
                if expansions.canRevert(controller) {
                    Button("\(undoLabel) · Undo") { expansions.revert(controller) }
                        .buttonStyle(.plain)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
        }
    }

    /// A clip model is not making "a picture", and saying so is the cheapest
    /// way to tell someone what they are about to get.
    private func placeholder(_ recipe: GenerationRecipe) -> String {
        recipe.temporal == nil ? "Describe a picture…" : "Describe a clip…"
    }

    /// Absence of `sourceImage` means YES -- raw `sourceImage?.isSupported` had it backwards.
    ///
    /// The recipe's own permission only; whether the well is DRAWN is
    /// `ImageConditioningWells.layout`'s decision, which folds this together
    /// with the reference relation.
    static func showsSourceWell(for recipe: GenerationRecipe) -> Bool { recipe.capabilities.readsSourceImage }

    /// What `ExpandStore.canRevert`'s affordance says was just done to the
    /// prompt -- the operation the accepted choice actually carried out.
    private var undoLabel: String {
        draft.promptTransform?.operation == .remix ? "remixed" : "expanded"
    }

    private var promptAvailableHeight: CGFloat {
        // Other controls may scroll, but Generate always remains outside that area.
        max(0, scrollHeight - 100)
    }

    private var scrollHeight: CGFloat {
        // The action row stays outside the scroll area and inside the window.
        // 32 is the capsule padding; 12 is each VStack gap.
        let stepSpace = controller.run.steps == nil ? 0 : stepsHeight + 12
        return max(0, maxHeight - 32 - actionsHeight - 12 - stepSpace)
    }

    private func loadHistory() {
        cycler.setEntries(host.map { history.entries(on: $0.id).map(\.prompt) } ?? [])
    }

    private func historyKey(up: Bool) -> Bool {
        guard PromptEditingKeyboard.allowsHistory(editorOpen: isEditingPrompt, promptFocused: focusedField == .prompt),
              let editor = NSApp.keyWindow?.firstResponder as? NSTextView else { return false }
        let selection = editor.selectedRange()
        guard up ? PromptHistoryCaret.isOnFirstLine(draft.prompt, selection: selection)
                 : PromptHistoryCaret.isOnLastLine(draft.prompt, selection: selection)
        else { return false }
        let next = up ? cycler.previous(from: draft.prompt) : cycler.next(from: draft.prompt)
        guard let next else { return false }
        PromptHistoryRecall.apply(next, to: &draft)
        expansions.lastAcceptedPrompt = nil
        editor.setSelectedRange(NSRange(location: (next as NSString).length, length: 0))
        return true
    }

    private func installHistoryKeyMonitor() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            let modifiers = event.modifierFlags.intersection([.command, .option, .control, .shift])
            if isEditingPrompt { return event }
            guard modifiers.isEmpty else { return event }
            switch event.keyCode {
            case 126 where historyKey(up: true): return nil
            case 125 where historyKey(up: false): return nil
            default: return event
            }
        }
    }

    private func removeHistoryKeyMonitor() {
        if let keyMonitor { NSEvent.removeMonitor(keyMonitor) }
        keyMonitor = nil
    }
}
