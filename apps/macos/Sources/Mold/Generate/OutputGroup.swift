import MoldClient
import MoldStyle
import SwiftUI

/// Format, upscaler, and whether this print is kept.
struct OutputGroup: View {
    let output: OutputCapabilities?
    /// This host's whole model list, upscalers included -- filtered here by
    /// `UpscaleRow.resolve`, never assumed by name.
    let models: [Model]
    @Binding var draft: RenderDraft

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            formatSection
            transparencySection
            upscaleSection
            Toggle("Save to library", isOn: $draft.savesToGallery)
                .help("Keep new results in the Library. When off, they go to Trash.")
            if !draft.savesToGallery {
                Text("New prints go to Trash and remain recoverable until the trash is purged.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    @ViewBuilder private var formatSection: some View {
        switch Row.resolve(output: output) {
        case .hidden:
            EmptyView()
        case let .fixed(name, reason):
            LabeledSection("Format") {
                Text(name)
                if let reason {
                    Text(reason).font(.caption).foregroundStyle(.secondary)
                }
            }
        case let .picker(formats, defaultFormat):
            VStack(alignment: .leading, spacing: 5) {
                InspectorField("Format") {
                    Picker("Format", selection: formatBinding(fallback: defaultFormat)) {
                        ForEach(formats, id: \.self) { format in
                            Text(format.uppercased()).tag(format)
                                // JPEG has no alpha channel: admission refuses the
                                // pair rather than flattening the cut-out.
                                .disabled(draft.transparencyBlocksFormat(format))
                        }
                    }
                    .help("Choose the file format for new results")
                    .labelsHidden()
                    .frame(maxWidth: .infinity)
                }
                if formats.contains(where: draft.transparencyBlocksFormat) {
                    Text(TransparencyControl.unavailableFormatReason)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if output?.audioRequiresMp4 == true {
                    Text("Audio requires MP4. Choosing another format turns audio off.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    /// Drawn only where the recipe advertises an ADJUSTABLE block -- an
    /// older host, a hidden recipe and one with no alpha format all draw
    /// nothing (`TransparencyCapability.control`). The choice stays on the
    /// draft either way and travels only while this row is drawn.
    @ViewBuilder private var transparencySection: some View {
        if Self.offersTransparency(draft) {
            VStack(alignment: .leading, spacing: 4) {
                Toggle(TransparencyControl.label, isOn: transparencyBinding)
                    .help("Create an image with a transparent background using a supported file format")
                Text(TransparencyControl.note)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// Turning it on moves a JPEG pick to the first alpha format.
    private var transparencyBinding: Binding<Bool> {
        Binding(
            get: { draft.transparentBackground },
            set: { draft = draft.settingTransparentBackground($0, output: output) }
        )
    }

    @ViewBuilder private var upscaleSection: some View {
        let ready = UpscaleRow.resolve(models: models)
        if !ready.isEmpty {
            InspectorField("Upscale") {
                Picker("Upscale", selection: $draft.upscaleModel) {
                    Text("Don't upscale").tag(String?.none)
                    ForEach(ready) { model in
                        Text(model.headline).tag(String?.some(model.name))
                    }
                }
                .labelsHidden()
                .frame(maxWidth: .infinity)
            }
            .help("""
            Make the result larger. Images are enlarged before delivery; clips \
            are enlarged in a second job after they reach the Library.
            """)
        }
    }

    /// Seeds the picker's display from the recipe's default WITHOUT writing
    /// it into the draft: the getter falls back to `fallback`, so an
    /// untouched draft keeps `outputFormat` absent and the server's own
    /// default applies. Only a real pick reaches the setter.
    private func formatBinding(fallback: String) -> Binding<String> {
        Binding(
            get: { draft.outputFormat ?? fallback },
            set: { draft = draft.selectingOutputFormat($0, output: output) }
        )
    }
}

extension OutputGroup {
    /// Whether the Transparent background row is drawn: the adopted recipe's
    /// own `transparencyControl`, recorded on the draft by `adopting`.
    static func offersTransparency(_ draft: RenderDraft) -> Bool {
        draft.transparency != nil
    }

    /// What the Format row shows, resolved purely from the recipe's own
    /// block -- no view needed to test it.
    enum Row: Equatable {
        case hidden
        case fixed(name: String, reason: String?)
        case picker(formats: [String], defaultFormat: String)

        static func resolve(output: OutputCapabilities?) -> Row {
            guard let output else { return .hidden }
            guard !output.isFixed else {
                let name = (output.formats.first ?? output.defaultFormat).uppercased()
                return .fixed(name: name, reason: output.deliveryReason)
            }
            return .picker(formats: output.formats, defaultFormat: output.defaultFormat)
        }
    }
}

/// Installed and ready upscalers on this machine -- never a hard-coded name
/// like `real-esrgan-x4plus`.
enum UpscaleRow {
    static func resolve(models: [Model]) -> [Model] {
        models.filter { $0.isUpscaler && $0.isReady }
    }
}
