import Foundation
import Testing
@testable import MoldClient

private func boundaryCapabilities(_ wire: String, required: Bool = false) throws -> RecipeCapabilities {
    try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data("""
    {"boundary_frames":{"mode":"adjustable","wire":"\(wire)","min_frames":9,
      "first_required":\(required),"last_required":false}}
    """.utf8))
}

private func boundaryPicture(_ name: String) -> ImportedPicture {
    ImportedPicture(encoded: name, name: name, data: Data())
}

@Test func wanEndpointsMoveWithClipLengthAndOmitSource() throws {
    let caps = try boundaryCapabilities("wan-pair")
    var draft = RenderDraft()
    draft.frames = 121
    draft.media.sourceImage = "STALE"
    BoundaryFramePolicy.set(first: true, picture: boundaryPicture("FIRST"), draft: &draft, capabilities: caps)
    #expect(BoundaryFramePolicy.refusal(draft: draft, capabilities: caps) != nil)
    BoundaryFramePolicy.set(first: false, picture: boundaryPicture("LAST"), draft: &draft, capabilities: caps)
    #expect(draft.media.sourceImage == nil)
    #expect(draft.media.keyframes.map(\.frame) == [0, 120])
    #expect(BoundaryFramePolicy.refusal(draft: draft, capabilities: caps) == nil)
    draft.frames = 5
    BoundaryFramePolicy.apply(to: &draft, capabilities: caps)
    #expect(draft.frames == 9)
    #expect(draft.media.keyframes.map(\.frame) == [0, 8])
    #expect(draft.media.keyframes.map(\.image) == ["FIRST", "LAST"])
}

@Test func h3EndpointsUseSourceAndOptionalClosingAnchor() throws {
    let caps = try boundaryCapabilities("h3-endpoints", required: true)
    var draft = RenderDraft()
    draft.frames = 107
    #expect(BoundaryFramePolicy.refusal(draft: draft, capabilities: caps) != nil)
    BoundaryFramePolicy.set(first: true, picture: boundaryPicture("FIRST"), draft: &draft, capabilities: caps)
    #expect(BoundaryFramePolicy.refusal(draft: draft, capabilities: caps) == nil)
    BoundaryFramePolicy.set(first: false, picture: boundaryPicture("LAST"), draft: &draft, capabilities: caps)
    #expect(draft.media.sourceImage == "FIRST")
    #expect(draft.media.keyframes.map(\.frame) == [106])
    BoundaryFramePolicy.set(first: false, picture: nil, draft: &draft, capabilities: caps)
    #expect(draft.media.keyframes.isEmpty)
    #expect(draft.media.sourceImage == "FIRST")
}

@Test func unknownBoundaryProtocolsOfferNoInputs() throws {
    #expect(BoundaryFramePolicy.resolve(capabilities: try boundaryCapabilities("future")) == nil)
}

@Test func boundarySwitchesPreserveInterpolationIndicesAndIndependentAnchors() throws {
    let normal = try MoldJSON.decoder.decode(RecipeCapabilities.self,
        from: Data(#"{"keyframes":{"mode":"adjustable","required":false}}"#.utf8))
    let wan = try boundaryCapabilities("wan-pair")
    let h3 = try boundaryCapabilities("h3-endpoints")
    var draft = RenderDraft()
    draft.frames = 121
    draft.media.reconcile(for: normal)
    draft.media.keyframes = [KeyframeCondition(frame: 0, image: "A"),
        KeyframeCondition(frame: 40, image: "B"), KeyframeCondition(frame: 80, image: "C")]
    draft.media.reconcile(for: wan)
    BoundaryFramePolicy.apply(to: &draft, capabilities: wan)
    #expect(draft.media.keyframes.isEmpty)
    BoundaryFramePolicy.set(first: true, picture: boundaryPicture("WAN-FIRST"), draft: &draft, capabilities: wan)
    BoundaryFramePolicy.set(first: false, picture: boundaryPicture("WAN-LAST"), draft: &draft, capabilities: wan)
    draft.media.reconcile(for: h3)
    BoundaryFramePolicy.set(first: true, picture: boundaryPicture("H3-FIRST"), draft: &draft, capabilities: h3)
    BoundaryFramePolicy.set(first: false, picture: boundaryPicture("H3-LAST"), draft: &draft, capabilities: h3)
    draft.media.reconcile(for: normal)
    #expect(draft.media.keyframes.map(\.frame) == [0,40,80])
    #expect(draft.media.keyframes.map(\.image) == ["A","B","C"])
    draft.media.reconcile(for: wan)
    BoundaryFramePolicy.apply(to: &draft, capabilities: wan)
    #expect(draft.media.keyframes.map(\.image) == ["WAN-FIRST","WAN-LAST"])
    #expect(draft.media.sourceImage == nil)
    draft.media.reconcile(for: h3)
    BoundaryFramePolicy.apply(to: &draft, capabilities: h3)
    #expect(draft.media.keyframes.map(\.image) == ["H3-LAST"])
    #expect(draft.media.sourceImage == "H3-FIRST")
}
