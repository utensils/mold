import MoldClient
import SwiftUI

/// A spacious, live view of the same draft used by the compact composer.
struct PromptEditorSheet: View {
    @Binding var draft: RenderDraft
    let recipe: GenerationRecipe
    let host: MoldHost?
    @Binding var destination: Destination
    @Environment(\.dismiss) private var dismiss
    @Environment(PromptHistoryStore.self) private var history
    @Environment(ExpandStore.self) private var expansions
    @Environment(GenerateController.self) private var controller
    @FocusState private var textFocused: Bool
    @State private var editing = PromptEditingSession()
    @State private var clearedRewrite: LastAcceptedPrompt?
    @State private var showsHistory = false
    @State private var search = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Text("Prompt").font(.title2.weight(.semibold))
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.escape, modifiers: [])
                    .accessibilityIdentifier("prompt-editor-done")
            }
            TextEditor(text: $draft.prompt)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(12)
                .background(.background, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                .frame(minHeight: 340, maxHeight: .infinity)
                .focused($textFocused)
                .overlay(alignment: .topLeading) {
                    if draft.prompt.isEmpty {
                        Text(recipe.temporal == nil ? "Describe a picture…" : "Describe a clip…")
                            .foregroundStyle(.secondary)
                            .padding(.leading, 17).padding(.top, 13)
                            .allowsHitTesting(false).accessibilityHidden(true)
                    }
                }
                .accessibilityLabel("Prompt")
                .accessibilityIdentifier("prompt-editor-text")
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 16) {
                    historyActions
                    Spacer()
                    rewriteActions
                }
                VStack(alignment: .leading, spacing: 12) {
                    HStack(spacing: 16) {
                        historyActions
                        Spacer()
                    }
                    HStack(spacing: 16) {
                        rewriteActions
                        Spacer()
                    }
                }
            }
            .buttonStyle(.borderless)
        }
        .padding(28)
        .frame(minWidth: 580, idealWidth: 760, minHeight: 500, idealHeight: 640)
        .focusedValue(\.editingText, true)
        .onAppear { textFocused = true }
        .onChange(of: draft.prompt) { _, prompt in
            editing.observe(prompt: prompt)
            if !prompt.isEmpty { clearedRewrite = nil }
        }
        .task(id: host?.id) {
            guard let host else { return }
            await history.refresh(on: host.id)
        }
    }

    private var historyActions: some View {
        HStack(spacing: 16) {
            Button("Recent prompts", systemImage: "clock") { showsHistory.toggle() }
                .disabled(host == nil)
                .popover(isPresented: $showsHistory) { recentPrompts }
            Button("Clear", systemImage: "xmark") {
                clearedRewrite = expansions.lastAcceptedPrompt
                editing.clear(&draft)
                expansions.lastAcceptedPrompt = nil
                textFocused = true
            }
            .disabled(draft.prompt.isEmpty)
            if editing.canUndoClear(prompt: draft.prompt) {
                Button("Undo clear") {
                    editing.undoClear(&draft)
                    expansions.lastAcceptedPrompt = clearedRewrite
                    clearedRewrite = nil
                    textFocused = true
                }
            }
        }
    }

    private var rewriteActions: some View {
        HStack(spacing: 16) {
            if let host {
                PromptWand(recipe: recipe, host: host, draft: $draft, destination: $destination)
            }
            if expansions.canRevert(controller) {
                Button("Undo rewrite") { expansions.revert(controller) }
            }
        }
    }

    private var recentPrompts: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent prompts").font(.headline)
            TextField("Search recent prompts", text: $search)
                .textFieldStyle(.roundedBorder)
            if let host {
                if let failure = history.failureByHost[host.id] {
                    Text(failure).foregroundStyle(.secondary)
                    Button("Try again") { Task { await history.refresh(on: host.id) } }
                } else if history.unavailable.contains(host.id) {
                    Text("Prompt history is unavailable on this machine.")
                        .foregroundStyle(.secondary)
                } else if !history.hasLoaded(on: host.id) {
                    ProgressView("Loading recent prompts…")
                } else {
                    let page = RecentPromptPage(
                        entries: history.entries(on: host.id), query: search, limit: 50)
                    if page.visible.isEmpty {
                        Text(search.isEmpty ? "No recent prompts." : "No matching prompts.")
                            .foregroundStyle(.secondary)
                    }
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(page.visible) { entry in
                                Button {
                                    editing.replaced()
                                    PromptHistoryRecall.apply(entry.prompt, to: &draft)
                                    expansions.lastAcceptedPrompt = nil
                                    showsHistory = false
                                    textFocused = true
                                } label: {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(entry.prompt).lineLimit(4)
                                        Text(entry.model).font(.caption).foregroundStyle(.secondary)
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .contentShape(Rectangle())
                                }
                                .buttonStyle(.plain)
                                Divider()
                            }
                        }
                    }
                    .frame(maxHeight: 320)
                }
            }
        }
        .padding(20)
        .frame(width: 420)
    }
}
