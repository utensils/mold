import MoldClient
import SwiftUI

/// Re-fits the source picture whenever the canvas, the policy or the picture
/// itself moves.
///
/// The draft holds exactly what will be sent, which is this app's rule for
/// every byte field -- so the FIT is applied here rather than at submit, where
/// `RenderDraft.request` is a pure function with no pixels in it. The
/// unfitted copy stays in `sourceImageOriginal`, because fitting an
/// already-fitted picture crops a crop.
struct SourceFitTask: ViewModifier {
    @Binding var draft: RenderDraft

    /// Everything the fitted bytes depend on. A `.task(id:)` over this runs
    /// again exactly when one of them moves and never otherwise -- which is
    /// also what keeps a painted mask: while this does not run, nothing has
    /// invalidated it.
    struct Key: Equatable {
        let original: String?
        let width: Int
        let height: Int
        let policy: SourceFit
    }

    private var key: Key {
        Key(original: draft.media.sourceImageOriginal, width: draft.width,
            height: draft.height, policy: draft.media.sourceFit)
    }

    /// What was last actually applied. `.task(id:)` also fires on APPEAR --
    /// collapsing and re-expanding the inspector destroys and recreates this
    /// view -- so without it an idle re-render re-ran the fit, and the mask
    /// composition below had nothing to do but would still have rewritten a
    /// draft nobody touched.
    @State private var applied: Key?

    func body(content: Content) -> some View {
        content.task(id: key) {
            guard applied != key else { return }
            await refit()
        }
    }

    private func refit() async {
        guard let encoded = draft.media.sourceImageOriginal,
              let original = Data(base64Encoded: encoded),
              draft.width > 0, draft.height > 0
        else { return }
        let target = (width: draft.width, height: draft.height)
        let policy = draft.media.sourceFit
        let fitted = await SourceFitRender.fit(
            original, name: draft.media.sourceImageOriginalName ?? "source.png",
            target: target, policy: policy)
        guard !Task.isCancelled else { return }
        // The key has to still describe this draft: a canvas moved while the
        // fit was in flight would otherwise land the OLD crop over the new one.
        guard key.original == encoded, key.width == target.width,
              key.height == target.height, key.policy == policy else { return }

        if let fitted {
            draft.media.sourceImage = fitted.encoded
            draft.media.sourceImageName = fitted.name
        } else {
            // Nothing to do: the picture already fills the canvas exactly.
            draft.media.sourceImage = encoded
            draft.media.sourceImageName = draft.media.sourceImageOriginalName
        }
        // What was painted, plus whatever bands this fit added -- never
        // instead of it (`SourceFitRender+Mask`).
        guard let transform = SourceFitRender.transform(
            of: original, target: target, policy: policy) else { return }
        let composed = await SourceFitRender.mask(
            existing: draft.media.maskImage.flatMap { Data(base64Encoded: $0) },
            transform: transform)
        guard !Task.isCancelled, key.original == encoded, key.policy == policy else { return }
        if let composed { draft.media.maskImage = composed.base64EncodedString() }
        applied = key
    }
}

extension View {
    /// Keeps `draft.media.sourceImage` the fitted form of
    /// `sourceImageOriginal`, off the main actor.
    func refittingSource(draft: Binding<RenderDraft>) -> some View {
        modifier(SourceFitTask(draft: draft))
    }
}
