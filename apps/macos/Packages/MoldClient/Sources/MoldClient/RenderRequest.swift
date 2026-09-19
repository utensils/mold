import Foundation

/// A draft, turned into what one press of Generate submits.
///
/// Its own namespace rather than more of `RenderDraft`: the draft is what is
/// being AUTHORED -- a prompt, a canvas, the wells and the controls -- and
/// this is the translation to the wire, which answers questions the draft
/// itself does not hold an opinion on: which conditioning well ships when a
/// recipe keeps both, what the DESTINATION host understands about identity
/// photos, and how a batch of four becomes four independent one-output
/// requests. `RenderDraft(reusing:)` is the journey back, and it stays on the
/// draft because what it produces is a draft.
public enum RenderRequest {
    /// One request, with `batchSize` always 1.
    ///
    /// `/api/generation-batches` refuses any child whose `batch_size` is not
    /// 1 (`crates/mold-server/src/queue_media_admission.rs:380-386`), so this
    /// never reads the draft's own `batchSize` -- see `batch(_:model:copies:
    /// randomBase:)` for how a batch of several is actually built.
    ///
    /// `maxIdentityPhotos` is `Capabilities.maxIdentityPhotos` for the host
    /// this request is going to -- it decides `id_image` vs `id_images`
    /// (`IdentityConditioning.wire(maxPhotos:)`) and has no honest default,
    /// so it defaults to 0 rather than guessing a host's capability.
    public static func one(
        _ draft: RenderDraft, model: String, maxIdentityPhotos: Int = 0
    ) -> GenerateRequest {
        var request = GenerateRequest(
            prompt: draft.prompt, model: model, width: draft.width, height: draft.height,
            steps: draft.steps, guidance: draft.guidance, batchSize: 1,
            negativePrompt: draft.negativePrompt.isEmpty ? nil : draft.negativePrompt,
            seed: draft.locksSeed ? draft.seed : nil
        )
        request.frames = draft.frames
        request.fps = draft.fps
        request.pipeline = draft.pipeline
        // A controllable video recipe receives BOTH choices explicitly:
        // omission resolves to audio-on at the server, so an authored OFF
        // must travel as false. Fixed-audio H3 and t2a recipes omit the flag;
        // an LTX checkpoint missing audio assets must send false, while
        // unrelated families omit the flag.
        if draft.requiresAudio {
            request.enableAudio = nil
        } else if draft.usesOptionalAudioBranch || draft.supportsAudio {
            request.enableAudio = draft.enableAudio
        }
        if draft.supportsAudio && !draft.requiresAudio {
            request.videoOnly = VideoOnlyPolicy.requestValue(
                enabled: draft.videoOnly, draft.videoOnlyInputs)
        }
        // WHICH well ships is `requestConditioning`'s decision, never
        // "references if there are any": an EXCLUSIVE recipe keeps both wells
        // and one render carries a source image OR references, so a builder
        // that emitted both is refused (`sourceMediaPlan.ts:217-241`). An
        // extend outranks both -- it pins the first frames from the source
        // clip's own tail, and a source going stale between two independent
        // controls must never ride out beside it.
        let carries = draft.media.requestConditioning
        let carriesSource = carries.carriesSource && draft.media.extendVideo == nil
        request.sourceImage = carriesSource ? draft.media.sourceImage : nil
        request.sourceImageName = carriesSource ? draft.media.sourceImageName : nil
        request.editImages = carries.carriesReferences ? draft.media.editImages : nil
        request.referenceWeight = carries.carriesReferences ? draft.media.referenceWeight : nil
        // Strength means nothing with nothing to apply it to, and a mask with
        // no source is refused outright (`validation.rs:3101-3107`).
        // MiniMax H3 validates its fixed value even for Ref2VA, where the
        // request carries references and no source image. More generally an
        // advertised false means there is no authored denoise control: put
        // the protocol's neutral/fixed value on the wire. Older hosts (`nil`)
        // retain the pre-profile source-only behaviour.
        request.strength = switch draft.supportsStrength {
        case false: 1
        case true: draft.strength
        case nil: carriesSource ? draft.strength : nil
        }
        request.maskImage = carriesSource ? draft.media.maskImage : nil
        request.loras = draft.media.loras.isEmpty ? nil : draft.media.loras
        applyIdentity(draft, to: &request, maxPhotos: maxIdentityPhotos)
        applyControl(draft, to: &request)
        applyAdvanced(draft, to: &request)
        applySourceFit(draft, to: &request, carriesSource: carriesSource)
        applyClip(draft, to: &request)

        let trimmedTitle = draft.title.trimmingCharacters(in: .whitespacesAndNewlines)
        request.title = trimmedTitle.isEmpty ? nil : trimmedTitle
        // Folds the title into the tag list when the switch asks for it --
        // see `ClientTags.compose`.
        let composedTags = ClientTags.compose(
            explicit: draft.tags, title: request.title, autoTagTitle: draft.autoTagTitle
        ).tags
        request.tags = composedTags.isEmpty ? nil : composedTags
        request.collection = draft.collectionName.map(CollectionRef.named)
        request.outputFormat = draft.outputFormat
        request.upscaleModel = draft.upscaleModel
        // Absent means the server's own default (save). `false` is the only
        // instruction worth sending over the wire.
        request.saveToGallery = draft.savesToGallery ? nil : false
        request.originalPrompt = draft.originalPrompt
        request.promptTransform = draft.promptTransform
        return request
    }

    /// Fills in the identity fields from `identity`, choosing `id_image` vs
    /// `id_images` from what the host understands. `idWeight`/`idStartStep`
    /// ride only alongside a photograph (`identity.rs:988`), and the start
    /// step is clamped BELOW the draft's own step count -- the identity and
    /// steps controls live in different places on screen, so dragging Steps
    /// down after Start step was set must not silently arm a 422
    /// (`identity.rs:560-566`).
    private static func applyIdentity(
        _ draft: RenderDraft, to request: inout GenerateRequest, maxPhotos: Int
    ) {
        guard let identity = draft.media.identity,
              let wire = identity.wire(maxPhotos: maxPhotos) else { return }
        switch wire {
        case let .single(photo):
            request.idImage = photo.encoded
            request.idImageName = photo.name
        case let .several(photos):
            request.idImages = photos.map(\.encoded)
            request.idImageNames = photos.map(\.name)
        }
        request.idWeight = identity.weight
        request.idStartStep = Swift.min(identity.startStep, Swift.max(draft.steps - 1, 0))
    }
}
