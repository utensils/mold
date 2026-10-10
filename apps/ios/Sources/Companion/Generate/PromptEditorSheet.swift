import MoldClient
import SwiftUI

/// A spacious native editor of the authoritative draft. Every dismissal retains edits.
struct PromptEditorSheet: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    @Environment(\.dynamicTypeSize) private var size
    @FocusState private var editing: Bool
    @Binding var expansionUndo: PromptExpansionUndo
    @Binding var expandingPrompt: Bool
    @Binding var promptSuggestions: [String]
    @State private var session = PromptEditingSession()
    @State private var clearedExpansion: ClearedPromptExpansion?
    @State private var historyHost: MoldHost?

    var body: some View {
        NavigationStack {
            TextEditor(text: Binding(get: { generate.draft.prompt }, set: { text in
                session.replaced()
                clearedExpansion = nil
                generate.draft.prompt = text
            }))
            .overlay(alignment: .topLeading) {
                if generate.draft.prompt.isEmpty {
                    Text("Describe what you want to make…")
                        .foregroundStyle(.secondaryText)
                        .padding(.top, 8).padding(.leading, 5)
                        .allowsHitTesting(false).accessibilityHidden(true)
                }
            }
            .font(.body)
            .focused($editing)
            .accessibilityLabel("Prompt")
            .accessibilityIdentifier("prompt-editor-text")
            .scrollDismissesKeyboard(.interactively)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(16)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                Group {
                    if size.isAccessibilitySize {
                        actionsMenu
                    } else {
                        ViewThatFits(in: .horizontal) {
                            HStack(spacing: 12) {
                                actions.fixedSize()
                                expandButton.fixedSize()
                            }
                                .buttonStyle(.bordered)
                                .controlSize(.regular)
                            actionsMenu
                        }
                    }
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(uiColor: .systemBackground))
            }
            .navigationTitle("Prompt")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { editing = false; generate.saveDraft(); dismiss() }
                        .accessibilityIdentifier("prompt-editor-done")
                }
            }
            .task { editing = true }
            .onChange(of: generate.draft) { _, draft in
                session.observe(prompt: draft.prompt)
                if let clearedExpansion, !clearedExpansion.matches(draft) { self.clearedExpansion = nil }
            }
        }
        .sheet(item: $historyHost, onDismiss: { editing = true }) { host in
            PromptHistorySheet(host: host, hosts: hosts) { text in
                session.replaced()
                clearedExpansion = nil
                expansionUndo = PromptExpansionUndo()
                promptSuggestions = []
                PromptHistoryStore.recall(text, into: &generate.draft)
            }
        }
        .sheet(isPresented: Binding(get: { historyHost == nil && !promptSuggestions.isEmpty }, set: {
            if !$0 && historyHost == nil { promptSuggestions = [] }
        }), onDismiss: { editing = true }) {
            SuggestionsSheet(suggestions: promptSuggestions) { chosen in
                clearedExpansion = nil
                session.replaced()
                expansionUndo.record(original: generate.draft, expanded: chosen)
                generate.draft.prompt = chosen
                promptSuggestions = []
            }
        }
        .presentationDetents([.large])
        .presentationDragIndicator(.visible)
        .onDisappear { editing = false }
    }

    private var actionsMenu: some View {
        Menu("Prompt actions", systemImage: "ellipsis.circle") {
            actions
            ExpandButton(undo: $expansionUndo, working: $expandingPrompt,
                         suggestions: $promptSuggestions, menuItemsOnly: true)
        }
            .buttonStyle(.bordered)
            .accessibilityIdentifier("prompt-editor-actions")
    }

    private var expandButton: ExpandButton {
        ExpandButton(undo: $expansionUndo, working: $expandingPrompt, suggestions: $promptSuggestions)
    }

    @ViewBuilder private var actions: some View {
        Button("Recent prompts", systemImage: "clock.arrow.circlepath") {
            editing = false
            historyHost = generate.target ?? hosts.preferredHost
        }
        .disabled(generate.target == nil && hosts.preferredHost == nil)
        .accessibilityIdentifier("prompt-editor-history")
        if session.canUndoClear(prompt: generate.draft.prompt) {
            Button("Undo clear", systemImage: "arrow.uturn.backward") {
                session.undoClear(&generate.draft)
                expansionUndo = clearedExpansion?.restored(for: generate.draft) ?? PromptExpansionUndo()
                clearedExpansion = nil
            }
            .accessibilityIdentifier("prompt-editor-undo-clear")
        } else {
            Button("Clear", systemImage: "xmark.circle") {
                let undo = expansionUndo
                session.clear(&generate.draft)
                clearedExpansion = ClearedPromptExpansion(clearedDraft: generate.draft, undo: undo)
            }
            .disabled(generate.draft.prompt.isEmpty)
            .accessibilityIdentifier("prompt-editor-clear")
        }
    }
}
