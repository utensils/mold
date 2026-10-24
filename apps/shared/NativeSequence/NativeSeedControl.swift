import MoldClient
import SwiftUI

/// "Seed": `Random | Fixed`, and when Fixed an editable full-range number with
/// a dice button. One control, one wording, on both apps (`SeedChoice`).
struct NativeSeedControl: View {
    @Binding var draft: RenderDraft
    /// iOS forms name the row themselves; the Mac inspector's `ControlLabel`
    /// already does.
    var showsTitle = false
    var showsFootnote = false
    @State private var text = ""
    @FocusState private var editing: Bool

    private var mode: SeedChoice.Mode { SeedChoice.Mode.resolve(draft) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if showsTitle { Text("Seed") }
            if mode == .fixed {
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) { picker; number }
                    VStack(alignment: .leading, spacing: 6) { picker; number }
                }
            } else {
                picker
            }
            if showsFootnote {
                Text(SeedChoice.help).font(.caption).foregroundStyle(.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .onAppear { text = SeedChoice.text(for: draft) }
        .onChange(of: draft.seed) { if !editing { text = SeedChoice.text(for: draft) } }
        .onChange(of: editing) { if !editing { text = SeedChoice.text(for: draft) } }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("seed-control")
    }

    private var picker: some View {
        Picker("Seed", selection: Binding(get: { mode }, set: { SeedChoice.select($0, in: &draft) })) {
            ForEach(SeedChoice.Mode.allCases, id: \.self) { Text($0.title).tag($0) }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .fixedSize()
        .help(SeedChoice.help)
        .accessibilityIdentifier("seed-mode")
    }

    private var number: some View {
        HStack(spacing: 4) {
            // A hidden widest value sizes the field, so all twenty digits of a
            // full UInt64 stay visible at every text size.
            ZStack {
                Text(verbatim: String(UInt64.max)).monospacedDigit().padding(.horizontal, 8).hidden()
                TextField("Seed", text: $text)
                    .monospacedDigit()
                    .textFieldStyle(.roundedBorder)
                    .focused($editing)
                    .onChange(of: text) { SeedChoice.edit(text, in: &draft) }
                    .onSubmit { text = SeedChoice.text(for: draft) }
                    .accessibilityLabel("Starting seed")
                    .accessibilityIdentifier("seed-value")
#if os(iOS)
                    .keyboardType(.numberPad)
#else
                    .focusedValue(\.editingText, editing ? true : nil)
                    .help("The starting seed for the next render. A batch uses this number, then adds one for each additional result.")
#endif
            }
            .fixedSize()
            Button {
                SeedChoice.pickNewSeed(in: &draft)
                text = SeedChoice.text(for: draft)
            } label: {
                Label("New seed", systemImage: "dice")
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.borderless)
#if os(iOS)
            .frame(minWidth: 44, minHeight: 44)
#else
            .frame(minWidth: 28, minHeight: 28)
#endif
            .help(SeedChoice.newSeedHelp)
            .accessibilityIdentifier("seed-new")
        }
    }
}
