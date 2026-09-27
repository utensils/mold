import Foundation
import Testing

@testable import MoldClient

// Two more things a print records, read from the same capture
// (`Fixtures/provenance-hal9000.json`, fourteen verbatim rows from hal9000)
// through the same `Provenance` helper. Added for the Library inspector,
// which SHOWS this rather than reusing it.
//
// **Fails today**: `mesh_workflow` is not decoded at all, so a stage of a
// text-to-3-D run is indistinguishable from a hand-authored picture -- which
// is exactly the confusion `MeshWorkflowProvenance` exists to end -- and the
// identity photograph's LABEL has no reader beside `identityDigests`.

@Test func readsWhichThreeDWorkflowAStageBelongsTo() throws {
    // hal9000's own text-to-3-D run: this row is its FIRST stage, the picture
    // the mesh was later made from, published as an ordinary print.
    let stage = try #require(
        try Provenance.metadata("mold-qwen-image-q8-1789529980561.png").meshWorkflow)

    #expect(stage.mode == "text_to_mesh")
    #expect(stage.role == "generated_image")
    #expect(stage.stageIndex == 0)
    #expect(stage.jobId?.isEmpty == false)
    // And an ordinary print carries none, which is what absence means.
    #expect(try Provenance.metadata("mold-realistic-vision-v5-fp16-1784141403169.png")
        .meshWorkflow == nil)
}

@Test func readsTheIdentityPhotographsOwnLabel() throws {
    let face = try Provenance.metadata("mold-jibmix-flux-fp8-1788383514251~nsfw.png")

    #expect(face.identityPhotoNames == ["IMG_0730.png"])
    #expect(try Provenance.metadata("mold-qwen-image-q8-1789529980561.png")
        .identityPhotoNames.isEmpty)
}

/// Nothing on hal9000 has ever produced `id_image_names` -- a multi-photograph
/// print records the plural and a single-photograph one deliberately does not
/// -- so the fixture cannot pin the precedence, exactly as it cannot for
/// `identityDigests`. SYNTHETIC, and for the same reason.
@Test func severalPhotographsOutrankTheSingularLabel() {
    #expect(Synthetic.metadata(#""id_image_names":["a.png","b.png"]"#)
        .identityPhotoNames == ["a.png", "b.png"])
    #expect(Synthetic.metadata(
        #""id_image_name":"one.png","id_image_names":["a.png","b.png"]"#)
        .identityPhotoNames == ["a.png", "b.png"])
}
