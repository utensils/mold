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
    let estimate: String?
    let maximumHeight: CGFloat
    var inline = false
    var inlineAction = false
    @FocusState private var editing: Bool

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
                ExpandButton()
                Spacer()
                Button("Done") { editing = false }
            }
        }
    }

    private var content: some View {
        @Bindable var generate = generate
        return VStack(alignment: .leading, spacing: 12) {
            if inline {
                if size >= .xxLarge {
                    VStack(alignment: .leading, spacing: 12) {
                        KindMenu()
                        MachineMenu()
                    }
                } else {
                    HStack {
                        KindMenu()
                        Spacer(minLength: 8)
                        MachineMenu()
                    }
                }
            }
            if generate.recipe?.capabilities.promptRequirement != .ignored {
                HStack(alignment: .top, spacing: 8) {
                    TextField("Prompt", text: $generate.draft.prompt,
                              prompt: Text(generate.kind == .clip ? "Describe a clip…" : "Describe a picture…")
                                .foregroundStyle(.secondaryText), axis: .vertical)
                        // At the smallest text one line is too short a
                        // target to hit; two reserved lines are not.
                        .lineLimit((size <= .small ? 2 : 1) ... (size.isAccessibilitySize ? 3 : 6))
                        .focused($editing)
                        .accessibilityIdentifier("generation-prompt")
                        .id("prompt")
                        .frame(minHeight: 44)
                    ExpandButton().labelStyle(.iconOnly)
                }
            } else {
                Text("This model works from a picture, not a description.")
                    .foregroundStyle(.secondaryText)
            }
            ModelMenu()
            PictureWells()
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
            if !inline || inlineAction { GenerateRow(estimate: estimate) }
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

    static func stacks(at size: DynamicTypeSize, phone: Bool = false) -> Bool { phone || size >= .xxLarge }

    var body: some View {
        let stacked = Self.stacks(at: size, phone: UIDevice.current.userInterfaceIdiom == .phone)
        let layout = stacked ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
                             : AnyLayout(HStackLayout(spacing: 12))
        layout {
            if let estimate {
                Text(estimate).monospacedDigit().foregroundStyle(.secondaryText)
            }
            if !stacked { Spacer(minLength: 0) }
            Button { generate.generate() } label: {
                Label("Generate", systemImage: "wand.and.sparkles")
                    .frame(maxWidth: stacked ? .infinity : nil)
            }
            .prominentAction()
            .controlSize(.large)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(generate.blocker != nil)
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
                if let temporal = recipe.temporal {
                    LengthChip(temporal: temporal)
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
