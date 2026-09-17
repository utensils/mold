import MoldClient
import MoldStyle
import SwiftUI

/// The ControlNet rows: an adapter picker, a control picture well, and a
/// strength slider. Split from `RefineGroup.swift` purely for size; the well
/// itself is `ControlPictureWell.swift`.
extension RefineGroup {
    @ViewBuilder var controlNetSection: some View {
        switch ControlNetRow.resolve(control: recipe?.capabilities.controlNet, models: models) {
        case .hidden:
            EmptyView()
        case let .needsAdapter(reason):
            LabeledSection("ControlNet") {
                Text(reason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Get one…") { destination = .models }
            }
        case let .ready(installed):
            LabeledSection("ControlNet") {
                controlNetReady(installed)
            }
        }
    }

    @ViewBuilder private func controlNetReady(_ installed: [Model]) -> some View {
        Picker("Adapter", selection: controlModelBinding) {
            Text("None").tag(String?.none)
            ForEach(installed) { model in
                Text(model.headline).tag(String?.some(model.name))
            }
        }
        .labelsHidden()
        if draft.media.control?.model != nil {
            HStack(spacing: 8) {
                ControlPictureWell(draft: $draft)
                if draft.media.control?.image == nil {
                    Text("Pick a control picture too.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    SliderControl(name: "ControlNet strength", value: controlScaleBinding,
                                  range: Control.scaleRange, step: 0.05) {
                        Text(controlScaleBinding.wrappedValue, format: .number.precision(.fractionLength(2)))
                    }
                }
            }
        }
    }

    /// Picking "None" with no picture staged clears the field back to nil
    /// entirely -- an empty `ControlConditioning` is not a meaningful state
    /// to leave sitting in the draft.
    private var controlModelBinding: Binding<String?> {
        Binding(
            get: { draft.media.control?.model },
            set: { newValue in
                var control = draft.media.control ?? ControlConditioning()
                control.model = newValue
                draft.media.control = (control.image == nil && control.model == nil) ? nil : control
            }
        )
    }

    private var controlScaleBinding: Binding<Double> {
        Binding(
            get: { draft.media.control?.scale ?? Control.defaultScale },
            set: { draft.media.control?.scale = Swift.max($0, 0) }
        )
    }
}

/// Installed and ready ControlNet adapters on this machine, resolved purely
/// from the recipe's own `controlnet` block and what this host has --
/// `RecipeCapabilities.controlNet` has already filtered `hidden` blocks and
/// missing ones to `nil`, so a non-nil `control` here always means the
/// recipe is asking.
enum ControlNetRow {
    enum Resolution: Equatable {
        case hidden
        case needsAdapter(reason: String)
        case ready([Model])
    }

    static let defaultReason = "No ControlNet adapter is installed on this machine."

    static func resolve(control: AdapterControl?, models: [Model]) -> Resolution {
        guard let control else { return .hidden }
        let installed = models.filter { $0.isControlNet && $0.isReady }
        guard !installed.isEmpty else {
            return .needsAdapter(reason: control.reason ?? defaultReason)
        }
        return .ready(installed)
    }
}
