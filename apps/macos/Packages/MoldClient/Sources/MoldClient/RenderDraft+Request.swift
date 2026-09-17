import Foundation

// Turning a draft into what one press of Generate submits. Split from
// `RenderDraft+Recipe.swift` (which keeps `adopting` and `refusal`) purely
// for size.
public extension RenderDraft {
    /// One request, with `batchSize` always 1.
    ///
    /// `/api/generation-batches` refuses any child whose `batch_size` is not
    /// 1 (`crates/mold-server/src/queue_media_admission.rs:380-386`), so this
    /// never reads the draft's own `batchSize` -- see `requests(model:copies:
    /// randomBase:)` for how a batch of several is actually built.
    ///
    /// `maxIdentityPhotos` is `Capabilities.maxIdentityPhotos` for the host
    /// this request is going to -- it decides `id_image` vs `id_images`
    /// (`IdentityConditioning.wire(maxPhotos:)`) and has no honest default,
    /// so it defaults to 0 (no identity block understood) rather than
    /// guessing a host's capability. Nothing in `Sources/Mold` passes the
    /// real value yet -- see the S2 report for the callers S3+ must update.
    func request(model: String, maxIdentityPhotos: Int = 0) -> GenerateRequest {
        var request = GenerateRequest(
            prompt: prompt, model: model, width: width, height: height,
            steps: steps, guidance: guidance, batchSize: 1,
            negativePrompt: negativePrompt.isEmpty ? nil : negativePrompt,
            seed: locksSeed ? seed : nil
        )
        request.frames = frames
        request.fps = fps
        request.pipeline = pipeline
        request.enableAudio = enableAudio ? true : nil
        request.videoOnly = VideoOnlyPolicy.requestValue(enabled: videoOnly, videoOnlyInputs)
        // An extend is the strongest claimant on the request's first frames
        // (`RenderDraft+Recipe.swift`'s `adopting`); this belt matches
        // `idStartStep`'s own -- the source well and the extend well are two
        // independent controls, and a value going stale between them must
        // never reach the wire alongside the extend that outranks it.
        request.sourceImage = media.extendVideo == nil ? media.sourceImage : nil
        request.sourceImageName = media.extendVideo == nil ? media.sourceImageName : nil
        request.editImages = media.editImages.isEmpty ? nil : media.editImages
        request.referenceWeight = media.editImages.isEmpty ? nil : media.referenceWeight
        // Strength only means something with something to apply it to.
        request.strength = media.sourceImage == nil ? nil : strength
        // A mask with no source is refused outright (`validation.rs:3101-3107`).
        request.maskImage = media.sourceImage == nil ? nil : media.maskImage
        request.loras = media.loras.isEmpty ? nil : media.loras
        applyIdentity(to: &request, maxPhotos: maxIdentityPhotos)
        applyControl(to: &request)
        applyClip(to: &request)

        let trimmedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        request.title = trimmedTitle.isEmpty ? nil : trimmedTitle
        // Folds the title into the tag list when the switch asks for it --
        // see `ClientTags.compose`.
        let composedTags = ClientTags.compose(
            explicit: tags, title: request.title, autoTagTitle: autoTagTitle
        ).tags
        request.tags = composedTags.isEmpty ? nil : composedTags
        request.collection = collectionName.map(CollectionRef.named)
        request.outputFormat = outputFormat
        request.upscaleModel = upscaleModel
        // Absent means the server's own default (save). `false` is the only
        // instruction worth sending over the wire.
        request.saveToGallery = savesToGallery ? nil : false
        request.originalPrompt = originalPrompt
        request.promptTransform = promptTransform
        return request
    }

    /// The requests one press of Generate submits.
    ///
    /// A batch of N is N INDEPENDENT one-output requests, because the server
    /// refuses any child whose `batchSize` is not 1 -- the batch's size is
    /// the LENGTH of this array, capped by
    /// `queue.heterogeneous_batch_max_outputs`. They share one prompt, one
    /// filing and one logical `batchId`, and differ only by seed, so four
    /// presses of one button give four variations rather than four copies.
    /// This mirrors the fleet's other GUI exactly
    /// (`web/src/pages/CreatePage.vue:3457-3464`).
    ///
    /// `randomBase` is injected so the fan-out is a pure function a test can
    /// pin; production passes `UInt64.random(in: 0 ... UInt64(UInt32.max))`,
    /// the 32-bit range web draws from.
    func requests(
        model: String, copies: Int, randomBase: UInt64, maxIdentityPhotos: Int = 0
    ) -> [GenerateRequest] {
        guard copies > 1 else { return [request(model: model, maxIdentityPhotos: maxIdentityPhotos)] }
        let batchId = UUID().uuidString
        // An unlocked seed with copies > 1 mints `randomBase` client-side --
        // the alternative is N absent seeds, and a host that picks its own
        // seed per child would make the four siblings unrelated in a way
        // nobody can reproduce.
        let base = locksSeed ? (seed ?? randomBase) : randomBase
        return (0 ..< copies).map { index in
            var sibling = request(model: model, maxIdentityPhotos: maxIdentityPhotos)
            // Wrapping: a seed near `UInt64.max` is legal and a trap is not
            // an answer.
            sibling.seed = base &+ UInt64(index)
            sibling.batchId = batchId
            sibling.batchIndex = index + 1
            sibling.batchCount = copies
            return sibling
        }
    }

    /// The request to PREVIEW a fan-out with.
    ///
    /// The count goes in `PlacementRequest.copies` and the request itself
    /// stays a single output, or the host multiplies the two and previews
    /// sixteen pictures for a batch of four
    /// (`studio/api/generationPlacement.ts:339-347`).
    func placementRequest(model: String, maxIdentityPhotos: Int = 0) -> GenerateRequest {
        request(model: model, maxIdentityPhotos: maxIdentityPhotos)
    }

    /// Fills in the identity fields from `identity`, choosing `id_image` vs
    /// `id_images` from what the host understands. `idWeight`/`idStartStep`
    /// ride only alongside a photograph (`identity.rs:988`), and the start
    /// step is clamped BELOW the draft's own step count -- the identity and
    /// steps controls live in different places on screen, so dragging Steps
    /// down after Start step was set must not silently arm a 422
    /// (`identity.rs:560-566`).
    private func applyIdentity(to request: inout GenerateRequest, maxPhotos: Int) {
        guard let identity = media.identity, let wire = identity.wire(maxPhotos: maxPhotos) else { return }
        switch wire {
        case let .single(photo):
            request.idImage = photo.encoded
            request.idImageName = photo.name
        case let .several(photos):
            request.idImages = photos.map(\.encoded)
            request.idImageNames = photos.map(\.name)
        }
        request.idWeight = identity.weight
        request.idStartStep = Swift.min(identity.startStep, Swift.max(steps - 1, 0))
    }

    /// `control_image` and `control_model` are a symmetric pair
    /// (`validation.rs:3079-3090`): either alone is refused. The Refine
    /// group's picker and picture well can each be filled in before the
    /// other, so a draft with only one half sends NEITHER rather than a
    /// request the server would 422.
    private func applyControl(to request: inout GenerateRequest) {
        guard let image = media.control?.image, let model = media.control?.model else { return }
        request.controlImage = image
        request.controlModel = model
        request.controlScale = Swift.max(media.control?.scale ?? Control.defaultScale, 0)
    }
}
