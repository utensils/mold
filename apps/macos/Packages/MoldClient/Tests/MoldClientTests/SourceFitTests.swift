import Foundation
import Testing

@testable import MoldClient

/// The img2img fit policy, its geometry and its wire shape.
/// **Fails today**: nothing in this app fitted a source picture.
struct SourceFitTests {
    /// Crop takes the LARGER scale so the canvas is filled and the edges fall
    /// outside -- which is why the offsets are NEGATIVE.
    @Test func cropFillCoversTheCanvasAndTrimsSymmetrically() {
        let transform = SourceFitTransform.resolve(
            source: (2000, 1000), target: (1000, 1000), policy: .default)
        #expect(transform.drawWidth == 2000)
        #expect(transform.drawHeight == 1000)
        #expect(transform.offsetX == -500)
        #expect(transform.offsetY == 0)
        #expect(transform.maskPaddedPixels == false)
        #expect(transform.maskPadding.isEmpty)
    }

    @Test func aLeftAlignedCropTrimsOnlyTheRight() {
        let transform = SourceFitTransform.resolve(
            source: (2000, 1000), target: (1000, 1000),
            policy: .cropFill(alignX: .left, alignY: nil))
        #expect(transform.offsetX == 0)
    }

    /// Pad takes the SMALLER scale so the whole picture is kept, and the bands
    /// it adds are what a `pad-repaint` marks for the model to paint.
    @Test func padRepaintKeepsTheWholePictureAndNamesItsBands() {
        let transform = SourceFitTransform.resolve(
            source: (2000, 1000), target: (1000, 1000), policy: .padRepaint)
        #expect(transform.drawWidth == 1000)
        #expect(transform.drawHeight == 500)
        #expect(transform.offsetY == 250)
        #expect(transform.maskPadding == [
            SourceFitRect(x: 0, y: 0, width: 1000, height: 250),
            SourceFitRect(x: 0, y: 750, width: 1000, height: 250),
        ])
    }

    /// `pad-fit` adds the same bands and marks NONE of them: the borders stay.
    @Test func padFitAddsTheSameBandsAndMarksNoneOfThem() {
        let transform = SourceFitTransform.resolve(
            source: (2000, 1000), target: (1000, 1000), policy: .padFit)
        #expect(transform.drawHeight == 500)
        #expect(transform.maskPaddedPixels == false)
        #expect(transform.maskPadding.isEmpty)
    }

    @Test func stretchingIgnoresProportionsEntirely() {
        let transform = SourceFitTransform.resolve(
            source: (2000, 1000), target: (1000, 1000), policy: .lanczosResize)
        #expect(transform.drawWidth == 1000)
        #expect(transform.drawHeight == 1000)
        #expect(transform.isIdentity)
    }

    /// An `upscale-then-fit` defers to its own inner policy, and this app can
    /// still read one back even though it never authors one.
    @Test func anUpscaleThenFitIsGeometricallyItsInnerPolicy() {
        let policy = SourceFit.upscaleThenFit(upscalerModel: "real-esrgan", fit: .padFit)
        let transform = SourceFitTransform.resolve(
            source: (2000, 1000), target: (1000, 1000), policy: policy)
        #expect(transform.drawHeight == 500)
        #expect(transform.maskPaddedPixels == false)
    }

    /// The object's own keys are CAMELCASE, because studio's parser reads
    /// `alignX` and the server renames nothing inside an opaque value. A
    /// `CodingKeys` block would have been snake-cased by `MoldJSON.encoder`
    /// and studio would have read the alignment as absent.
    @Test func theWireObjectKeepsStudiosOwnSpelling() throws {
        var request = GenerateRequest(prompt: "p", model: "m", width: 8, height: 8,
                                      steps: 1, guidance: 1)
        request.sourceFit = .cropFill(alignX: .left, alignY: .bottom)
        let json = String(decoding: try MoldJSON.encoder.encode(request), as: UTF8.self)
        #expect(json.contains(#""source_fit":{"#))
        #expect(json.contains(#""alignX":"left""#))
        #expect(json.contains(#""alignY":"bottom""#))
    }

    @Test func everyPolicyRoundTripsThroughTheWire() throws {
        let policies: [SourceFit] = [
            .padRepaint, .padFit, .lanczosResize, .default,
            .cropFill(alignX: .right, alignY: .top),
            .upscaleThenFit(upscalerModel: "real-esrgan-x4plus:fp16", fit: .padFit),
        ]
        for policy in policies {
            let data = try MoldJSON.encoder.encode(policy)
            #expect(try MoldJSON.decoder.decode(SourceFit.self, from: data) == policy)
        }
    }

    /// Defensive, exactly as `parseSourceFitPolicy` is -- a corrupt or
    /// after-this-build value must never poison a live draft.
    @Test func nonsenseIsRefusedRatherThanDegraded() {
        let nested = #"{"mode":"upscale-then-fit","upscalerModel":"x","fit":"#
            + #"{"mode":"upscale-then-fit","upscalerModel":"y","fit":{"mode":"pad-fit"}}}"#
        for bad in [#"{"mode":"warp"}"#, #"{"mode":"crop-fill","alignX":"middle"}"#,
                    #"{"mode":"upscale-then-fit","fit":{"mode":"pad-fit"}}"#, nested] {
            #expect(throws: (any Error).self) {
                try MoldJSON.decoder.decode(SourceFit.self, from: Data(bad.utf8))
            }
        }
    }

    /// A fit describes what was done to bytes that ship. A render carrying no
    /// source carries no policy either.
    @Test func thePolicyRidesOnlyWithASourceThatShips() {
        var draft = RenderDraft()
        draft.media.sourceFit = .padFit
        #expect(draft.request(model: "m").sourceFit == nil)
        draft.media.sourceImage = "SRC"
        #expect(draft.request(model: "m").sourceFit == .padFit)
    }
}
