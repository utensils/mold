import MoldClient
import MoldStyle
import SwiftUI

/// How strongly the reference pictures pull.
///
/// IP-Adapter's strength, and the one control whose range, step and default
/// travel WITH the capability rather than being hard-coded per client:
/// `capabilities.reference_images.weight` is a `FloatControl` INSIDE the block
/// (`generation_profile.rs:470-484`), deliberately not a sibling bool the way
/// `supports_identity`'s bounds are. So this reads the advertised control and
/// never a constant of its own.
///
/// Absent block, absent `weight`, or a `hidden` mode: no control at all. An
/// older host that advertises references without a weight is exactly that --
/// older -- and the references still work, which is why absence hides this
/// one slider and refuses nothing.
struct ReferenceWeightControl: View {
    let control: FloatControl
    @Binding var draft: RenderDraft

    var body: some View {
        SliderControl(name: "Reference weight", value: value,
                      range: control.min ... control.max,
                      step: control.step) {
            Text(value.wrappedValue, format: .number.precision(.fractionLength(2)))
        }
        .rowActionMenu(GenerateMenus.referenceWeight(isAtDefault: isAtDefault),
                       perform: perform)
    }

    /// `nil` reads as the advertised default -- the draft records a weight
    /// only once somebody has moved it, so a request made without touching
    /// this slider sends nothing and the server applies its own default.
    private var value: Binding<Double> {
        Binding(get: { draft.media.referenceWeight ?? control.default },
                set: { draft.media.referenceWeight = control.clamp($0) })
    }

    private var isAtDefault: Bool { draft.media.referenceWeight == nil }

    private func perform(_ action: GenerateAction) {
        guard action == .resetReferenceWeight else { return }
        draft.media.referenceWeight = nil
    }
}

extension ReferenceWeightControl {
    /// The advertised weight control, or `nil` where this slider must not be
    /// drawn at all.
    ///
    /// Two questions, both of which have to be yes: does the recipe advertise
    /// a weight, and does the request this draft would build actually CARRY
    /// references. `requestConditioning`, never `editImages.isEmpty` -- on an
    /// EXCLUSIVE recipe a full strip can be the parked well, and a weight
    /// slider over conditioning that does not ship is furniture.
    static func resolve(
        recipe: GenerationRecipe?, model: Model?, media: DraftMedia
    ) -> FloatControl? {
        guard let recipe else { return nil }
        guard let references = recipe.capabilities.referenceImages(
            family: model?.family, model: model?.name) else { return nil }
        guard let weight = references.weight, weight.mode.isVisible else { return nil }
        guard media.requestConditioning.carriesReferences else { return nil }
        return weight
    }
}
