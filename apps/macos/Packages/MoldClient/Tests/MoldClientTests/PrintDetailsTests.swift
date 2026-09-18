import Foundation
import Testing

@testable import MoldClient

// What the inspector shows about one print.
//
// **Fails today**: there is no row model at all. `ProvenanceGrid` hard-codes
// eight rows -- machine, model, seed, steps, guidance, size, made, file -- so a
// clip says nothing about its rate or the picture it was made from, a mesh
// nothing about the grid it was extracted on, a sequence nothing about being
// several clips, and the scheduler, the adapters and the identity photograph
// are nowhere.
//
// Driven by `Fixtures/provenance-hal9000.json`, the same fourteen verbatim rows
// `ProvenanceTests` reads: between them a picture, a clip made from a source
// picture, a mesh, an authored sequence, an auto-chained one-shot, an upscale,
// a face and a 3-D workflow stage.

private let plato = UUID()

private func entry(_ filename: String) throws -> LibraryEntry {
    let print = try #require(try Provenance.prints().first { $0.filename == filename })
    return LibraryEntry(
        host: MoldHost(id: plato, name: "hal9000", baseURL: URL(string: "http://h")!),
        print: print)
}

private func groups(_ filename: String) throws -> [PrintDetailGroup] {
    PrintDetails.groups(for: try entry(filename))
}

private func value(_ groups: [PrintDetailGroup], _ group: String,
                   _ label: String) -> String? {
    groups.first { $0.title == group }?.rows.first { $0.label == label }?.value
}

private let picture = "mold-realistic-vision-v5-fp16-1784141403169.png"
private let clip = "mold-ltx-2.5-22b-distilled-q8-1789532686738.mp4"
private let mesh = "mold-hunyuan3d-mini-turbo-fp16-1788503140147.glb"
private let sequence =
    "mold-chain-8c352a5a9ac5d5c23549e66d96f07c97f26331a7798de3fce244cdc4da754073-take-1.mp4"
private let autoChained =
    "mold-chain-3cb05c3e9d8468260cb56042d6c579e243ae76ca7d6475ccaa1777c6ff49f581-take-1.mp4"
private let adapters = "mold-ltx-2-19b-distilled-fp8-1786851614872.mp4"
private let face = "mold-jibmix-flux-fp8-1788383514251~nsfw.png"
private let workflowStage = "mold-qwen-image-q8-1789529980561.png"

@Test func aPictureShowsWhatItRecordedAndNothingElse() throws {
    let rows = try groups(picture)

    #expect(rows.map(\.title) == ["Prompt", "Model", "Settings", "File"])
    #expect(value(rows, "Model", "Machine") == "hal9000")
    #expect(value(rows, "Settings", "Scheduler")?.isEmpty == false)
    #expect(value(rows, "File", "Format") == "PNG")
    // Nothing conditioned it and it does not move, so those headings are
    // ABSENT rather than a column of em-dashes.
    #expect(rows.contains { $0.title == "Clip" } == false)
    #expect(rows.contains { $0.title == "Made from" } == false)
    #expect(rows.contains { $0.title == "3-D" } == false)
}

/// A seed is shown so that somebody pastes it back, so it is never
/// group-separated.
@Test func aSeedIsCopyableDigitsRatherThanAFormattedNumber() throws {
    let seed = try #require(try entry(picture).print.metadata.seed)
    #expect(value(try groups(picture), "Settings", "Seed") == String(seed))
    #expect(value(try groups(picture), "Settings", "Seed")?.contains(",") == false)
}

@Test func aClipShowsItsRateAndThePictureItWasMadeFrom() throws {
    let rows = try groups(clip)

    #expect(value(rows, "Clip", "Frames") == "145")
    #expect(value(rows, "Clip", "Rate") == "24 fps")
    #expect(value(rows, "Clip", "Audio") == "On")
    #expect(value(rows, "Clip", "Pipeline") == "distilled")
    #expect(value(rows, "Made from", "Source picture")
        == "mold-qwen-image-q8-1789531868557.png")
    #expect(value(rows, "Settings", "Strength") == "0.75")
    #expect(value(rows, "File", "Took") == "3 min 58 sec")
}

@Test func aMeshShowsTheControlsThatRan() throws {
    let rows = try groups(mesh)

    #expect(value(rows, "3-D", "Detail") == "256³ grid")
    #expect(value(rows, "3-D", "Iso threshold") == "0.76")
    #expect(rows.contains { $0.title == "Clip" } == false)
}

/// An authored sequence and a one-shot the host had to render in pieces both
/// carry `chain`; only `output_mode` says which is which, and crediting
/// somebody with a split they never authored is the thing to avoid.
@Test func aSequenceSaysHowManyClipsItWasMadeOf() throws {
    #expect(value(try groups(sequence), "Sequence", "Made of")
        == "3 clips, one prompt each")
    #expect(value(try groups(sequence), "Sequence", "Sequence job")?.isEmpty == false)
    #expect(value(try groups(autoChained), "Rendered in clips", "Made of")?.isEmpty == false)
    #expect(try groups(autoChained).contains { $0.title == "Sequence" } == false)
    #expect(try groups(picture).contains { $0.title == "Sequence" } == false)
}

/// The recorded prompt is shown AS RECORDED -- a sequence's is every stage
/// newline-joined, and this is provenance. `firstStagePrompt` is reuse's
/// reduction, for authoring a new print, not for describing this one.
@Test func aSequencesPromptIsShownWholeRatherThanReduced() throws {
    let joined = try #require(try entry(sequence).print.metadata.prompt)
    #expect(joined.contains("\n"))
    #expect(value(try groups(sequence), "Prompt", "Prompt") == joined)
}

@Test func anAdapterReadsAsItsOwnNameAndItsStrength() throws {
    let stack = try #require(value(try groups(adapters), "Made from", "Adapter")
        ?? value(try groups(adapters), "Made from", "Adapters"))

    // Never the server-side path: the rest of it is a stranger's directory.
    #expect(stack.contains("/") == false)
    #expect(stack.contains(" at "))
}

@Test func aFacePrintNamesThePhotographAndItsStrength() throws {
    let rows = try groups(face)

    #expect(value(rows, "Made from", "Identity photo") == "IMG_0730.png")
    #expect(value(rows, "Made from", "Identity strength")?.isEmpty == false)
    #expect(value(rows, "Made from", "Identity from step")?.isEmpty == false)
}

@Test func aWorkflowStageSaysWhichRunItCameOutOf() throws {
    let rows = try groups(workflowStage)

    #expect(value(rows, "3-D workflow", "Workflow") == "Text to 3-D")
    #expect(value(rows, "3-D workflow", "Step") == "The picture it started from")
    #expect(try groups(picture).contains { $0.title == "3-D workflow" } == false)
}

@Test func theCanvasIsOnlyNamedWhenTheFileIsNotIt() throws {
    let upscaled = try groups("mold-real-esrgan-x4plus-fp16-1788488679298-upscaled.png")

    #expect(value(upscaled, "File", "Rendered at")?.isEmpty == false)
    #expect(value(upscaled, "Made from", "Made bigger with")?.isEmpty == false)
    // The clip rendered at the size it was delivered, so it says so once.
    #expect(value(try groups(clip), "File", "Rendered at") == nil)
}

@Test func aPromptIsProseAndAnEmptyOneIsNoRow() throws {
    let prompt = try #require(try groups(picture).first)

    #expect(prompt.title == "Prompt")
    #expect(prompt.rows.map(\.label) == ["Prompt"])
    #expect(prompt.rows.map(\.isProse) == [true])
}

/// Every row is copyable, and the reuse item is the Library's own -- filtered
/// out of the plan both other menus draw rather than written a second time.
@Test func everyRowOffersCopyingAndBorrowsUseTheseSettings() {
    let row = PrintDetailRow("Seed", "1789529636781972965")
    let plan = LibraryMenuPlan(scope: .prints, count: 1, canReuse: true)

    let menu = PrintDetails.menu(for: row, offering: plan.items)

    #expect(menu.map(\.title) == ["Copy Seed", "", "Use These Settings"])
    #expect(menu.first?.kind == .copy)
    #expect(menu.last?.kind == .library(.reuse))
    // A trashed print cannot be reused, and then there is only the one item.
    let trashed = LibraryMenuPlan(scope: .trash, count: 1, trashCount: 1)
    #expect(PrintDetails.menu(for: row, offering: trashed.items).map(\.kind) == [.copy])
}
