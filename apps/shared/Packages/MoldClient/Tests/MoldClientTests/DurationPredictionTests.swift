import Foundation
import Testing
@testable import MoldClient

struct DurationPredictionTests {
    @Test func preferenceRoundTripsAndReuseHonorsRequestedMode() throws {
        var draft = RenderDraft(); draft.frames = 121; draft.prefersAutomaticDuration = true
        let descriptor = DraftDescriptor(draft, model: "ltx", family: "ltx2", recipeID: nil)
        let decoded = try MoldJSON.decoder.decode(DraftDescriptor.self, from: MoldJSON.encoder.encode(descriptor))
        var restored = RenderDraft(); decoded.apply(to: &restored)
        #expect(restored.prefersAutomaticDuration && restored.frames == 121)
        let metadata = Synthetic.metadata(#""frames":121,"duration_prediction_requested":true"#)
        #expect(RenderDraft(reusing: metadata).prefersAutomaticDuration)
    }
    @Test func automaticDurationOmitsFramesAndParksOnUnsupportedModels() throws {
        var draft = RenderDraft()
        draft.frames = 121
        draft.prefersAutomaticDuration = true
        draft.supportsDurationPrediction = true
        #expect(RenderRequest.one(draft, model: "qualified").frames == nil)
        let wire = try #require(JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(RenderRequest.one(draft, model: "qualified"))) as? [String: Any])
        #expect(wire["frames"] == nil)
        draft.supportsDurationPrediction = false
        #expect(RenderRequest.one(draft, model: "older").frames == 121)
        #expect(draft.prefersAutomaticDuration)
        draft.supportsDurationPrediction = true
        draft.prefersAutomaticDuration = false
        #expect(RenderRequest.one(draft, model: "qualified").frames == 121)
    }
}
