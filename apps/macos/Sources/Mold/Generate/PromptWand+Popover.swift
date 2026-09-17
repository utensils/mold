import MoldClient
import SwiftUI

/// What the wand opens onto: a short list of alternatives to choose from, the
/// server's own advice when there is nothing to choose between, a plain
/// refusal, or a name and a way to get the model that would make this work.
struct PromptWandPopover: View {
    let expansion: Expansion
    let host: MoldHost
    @Binding var destination: Destination

    @Environment(GenerateController.self) private var controller
    @State private var selection: String?
    @FocusState private var listFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch expansion {
            case let .offering(offer):
                offering(offer)
            case let .advised(text):
                advice(text)
            case let .refused(message):
                refusal(message)
            case let .needsModel(model):
                needsModel(model)
            case .idle, .working:
                EmptyView()
            }
        }
        .padding(14)
        .frame(width: 320)
    }

    // MARK: - Choices

    private func offering(_ offer: Expansion.Offer) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title(for: offer)).font(.headline)
            Divider()
            List(offer.choices, selection: $selection) { choice in
                choiceRow(choice)
            }
            .listStyle(.plain)
            .frame(height: min(CGFloat(offer.choices.count) * 56, 220))
            .focused($listFocused)
            .onAppear { selection = offer.choices.first?.id; listFocused = true }
            .onKeyPress(.return) { acceptSelected(from: offer); return .handled }
            .onExitCommand { controller.dismissExpansion() }
            Divider()
            Text("Escape leaves the prompt as it was.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func choiceRow(_ choice: Expansion.Choice) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(choice.prompt).lineLimit(4)
            if !choice.dimensions.isEmpty {
                Text(choice.dimensions.map(\.rawValue).joined(separator: " · "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { controller.accept(choice) }
    }

    private func acceptSelected(from offer: Expansion.Offer) {
        guard let selection, let choice = offer.choices.first(where: { $0.id == selection }) else { return }
        controller.accept(choice)
    }

    private func title(for offer: Expansion.Offer) -> String {
        offer.choices.count == 3 ? "Three ways to say it" : "\(offer.choices.count) ways to say it"
    }

    // MARK: - Advice and refusal

    private func advice(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            ScrollView { Text(text).font(.callout) }.frame(maxHeight: 220)
            HStack {
                Spacer()
                Button("OK") { controller.dismissExpansion() }.keyboardShortcut(.defaultAction)
            }
        }
    }

    private func refusal(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(message).font(.callout)
            HStack {
                Spacer()
                Button("OK") { controller.dismissExpansion() }.keyboardShortcut(.defaultAction)
            }
        }
    }

    // MARK: - Needs a model

    private func needsModel(_ model: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("This machine would expand prompts locally, but hasn't pulled \(model) yet.")
                .font(.callout)
            HStack {
                Spacer()
                Button("Pull…") {
                    controller.dismissExpansion()
                    destination = .models
                }
                Button("OK") { controller.dismissExpansion() }.keyboardShortcut(.defaultAction)
            }
        }
    }
}
