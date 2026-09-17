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
            upscaleSection
            Toggle("Save to library", isOn: $draft.savesToGallery)
            Text("""
            Still rendered and still recoverable — it goes straight to Recently \
            Deleted, and the trash purges it.
            """)
            .font(.caption)
            .foregroundStyle(.secondary)
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
            LabeledSection("Format") {
                Picker("Format", selection: formatBinding(fallback: defaultFormat)) {
                    ForEach(formats, id: \.self) { format in
                        Text(format.uppercased()).tag(format)
                    }
                }
                .labelsHidden()
                if output?.audioRequiresMp4 == true {
                    Text("Audio-enabled delivery requires MP4.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder private var upscaleSection: some View {
        let ready = UpscaleRow.resolve(models: models)
        if !ready.isEmpty {
            LabeledSection("Upscale") {
                Picker("Upscale", selection: $draft.upscaleModel) {
                    Text("Don't upscale").tag(String?.none)
                    ForEach(ready) { model in
                        Text(model.headline).tag(String?.some(model.name))
                    }
                }
                .labelsHidden()
            }
            .help("""
            Images are upscaled before they arrive. A clip is upscaled after \
            it lands in the Library, as a second job.
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
            set: { draft.outputFormat = $0 }
        )
    }
}

extension OutputGroup {
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
