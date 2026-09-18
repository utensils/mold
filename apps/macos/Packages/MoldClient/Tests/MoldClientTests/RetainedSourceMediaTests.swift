import Foundation
import Testing

@testable import MoldClient

/// A print's retained private conditioning media: what the host answers, what
/// is worth saying about it, and which members are worth asking for.
///
/// `Fixtures/source-media-hal9000.json` holds three verbatim answers from
/// hal9000 (0.29.0, `GET /api/gallery/source-media/{filename}`, 2026-09-17),
/// one per shape this has to read.
///
/// **Fails today**: none of this exists -- the app never asks a host what it
/// kept, so Use These Settings restores a recipe with no picture in it.
struct RetainedSourceMediaTests {

    private func captured(_ key: String) throws -> RetainedSourceMedia.Inventory {
        struct Capture: Decodable {
            let sourceImage: Case
            let editImages: Case
            let legacy: Case
            struct Case: Decodable {
                let filename: String
                let body: RetainedSourceMedia.Inventory
            }
        }
        let capture = try MoldJSON.decoder.decode(
            Capture.self, from: RepoFixtures.fixture("source-media-hal9000.json"))
        switch key {
        case "source_image": return capture.sourceImage.body
        case "edit_images": return capture.editImages.body
        default: return capture.legacy.body
        }
    }

    @Test func readsAnAvailableInventoryAsTheHostSendsIt() throws {
        let inventory = try captured("source_image")
        #expect(inventory.availability == .available)
        let member = try #require(inventory.members.first)
        #expect(member.role == "source_image")
        #expect(member.displayName == "source_image-1")
        #expect(member.sizeBytes == 771_681)
        #expect(member.memberId.count == 64)
    }

    /// The commonest answer there is, and the one a required `members` would
    /// have failed on: the host omits the key entirely when the list is empty
    /// (`skip_serializing_if = "Vec::is_empty"`).
    @Test func anAnswerWithNoMembersAtAllStillDecodes() throws {
        let legacy = try captured("legacy")
        #expect(legacy.availability == .unavailableLegacy)
        #expect(legacy.members.isEmpty)
    }

    @Test func aStateThisBuildCannotNameIsNotAvailableAndSaysNothing() throws {
        let future = try MoldJSON.decoder.decode(
            RetainedSourceMedia.Inventory.self,
            from: Data(#"{"availability":"unavailable_quarantined"}"#.utf8))
        #expect(future.availability == .unknown)
        #expect(RetainedSourceMedia.disclosure(future.availability) == nil)
    }

    @Test func everySentenceIsTheOneTheOtherSurfacesUse() {
        #expect(RetainedSourceMedia.disclosure(.available) == nil)
        #expect(RetainedSourceMedia.disclosure(.unavailableLegacy)
            == "This older print did not retain its original source media. "
            + "Reattach it before developing.")
        #expect(RetainedSourceMedia.disclosure(.unavailableMissingOrCorrupt)
            == "This print\u{2019}s retained source media is missing or damaged. "
            + "Reattach it before developing.")
        #expect(RetainedSourceMedia.disclosure(.unavailableAuth)
            == "Connect this machine with an API key to restore its private source media.")
    }

    /// The rule that keeps a text-to-image print quiet. Its archive entry
    /// exists with no pins, which the host can only report as `legacy` -- so
    /// without this, every picture ever made would be told to reattach a
    /// source it never had.
    @Test func aTextToImagePrintIsNeverToldToReattachAnything() throws {
        let plain = try Provenance.metadata("mold-qwen-image-q8-1789529980561.png")
        #expect(RetainedSourceMedia.disclosable(plain) == false)
        #expect(RetainedSourceMedia.disclosable(nil) == false)
    }

    @Test func aPrintThatShippedConditioningBytesIsWorthASentence() throws {
        for filename in [
            "mold-ltx-2.5-22b-distilled-q8-1789532686738.mp4",      // source_image_sha256
            "mold-real-esrgan-x4plus-fp16-1788488679298-upscaled.png",  // edit_image_sha256s
            "mold-jibmix-flux-fp8-1788383514251~nsfw.png",           // id_image_sha256
            "mold-wan22-i2v-a14b-q5-1787718371258~nsfw.mp4",         // keyframes
            "mold-minimax-h3-ref2va-comfy-pruned-int8-1788197310484~nsfw.mp4",  // references
            "mold-ltx-2-19b-distilled-fp8-1787860285996~uat-ltx2-extend.mp4",   // overlap
        ] {
            #expect(RetainedSourceMedia.disclosable(try Provenance.metadata(filename)),
                    "\(filename) shipped conditioning bytes")
        }
    }

    /// `source_image_name` is a name with no bytes, and MiniMax H3 sets it
    /// alone -- so it is excluded on purpose, even though the local-stash
    /// restore does read it.
    @Test func aNameWithNoDigestIsNotEvidenceThatBytesShipped() {
        #expect(RetainedSourceMedia.disclosable(Synthetic.metadata("""
        "source_image_name":"a-picture.png"
        """)) == false)
    }

    // MARK: - Which members to ask for

    private func member(_ role: String, _ id: String = "m") -> RetainedSourceMedia.Member {
        RetainedSourceMedia.Member(memberId: id, role: role, displayName: role, sizeBytes: 1)
    }

    private var emptyRequest: GenerateRequest {
        GenerateRequest(prompt: "p", model: "m", width: 8, height: 8, steps: 4, guidance: 1)
    }

    @Test func asksForEveryRoleTheOutgoingRequestHasNoBytesFor() {
        let offered = ["source_image", "mask_image", "identity_image", "edit_images",
                       "control_image", "audio_file_path", "source_video", "extend_video",
                       "keyframes"].map { member($0, $0) }
        let wanted = RetainedSourceMedia.members(offered, forHydrating: emptyRequest)
        #expect(wanted.count == offered.count)
    }

    /// A picture somebody reattached by hand WINS. Asking for the retained
    /// one anyway is the host's own `RETAINED_MEDIA_REUSE_TARGET_CONFLICT`,
    /// and the whole reuse would be refused over a field the person filled in
    /// deliberately.
    @Test func neverAsksForARoleSomebodyAlreadyFilledInThemselves() {
        var reattached = emptyRequest
        reattached.sourceImage = "AAAA"
        reattached.editImages = ["BBBB"]
        let offered = [member("source_image", "a"), member("edit_images", "b"),
                       member("mask_image", "c")]
        #expect(RetainedSourceMedia.members(offered, forHydrating: reattached)
            .map(\.memberId) == ["c"])
    }

    /// A matted picture reused as input would be matted twice, so the host
    /// hands it over for DOWNLOAD and refuses it as generation input. Asking
    /// for one is a 422 for the whole reuse.
    @Test func neverAsksForProcessedMattingMedia() {
        let offered = [member("matting_processed_source_image", "a"),
                       member("matting_processed_references", "b"),
                       member("source_image", "c")]
        #expect(RetainedSourceMedia.members(offered, forHydrating: emptyRequest)
            .map(\.memberId) == ["c"])
    }

    @Test func neverAsksForARoleThisBuildCannotPlace() {
        // `references` included: this app models no H3 reference descriptors,
        // so there is nothing for retained reference bytes to attach to.
        let offered = [member("references", "a"), member("hdr_exr_dir", "b")]
        #expect(RetainedSourceMedia.members(offered, forHydrating: emptyRequest).isEmpty)
    }

    // MARK: - Cross-host relay

    @Test func inlinesEveryDownloadedMemberIntoTheOutgoingRequest() throws {
        let relayed = try RetainedSourceMedia.relayed([
            (member("source_image", "a"), Data([1, 2, 3])),
            (member("mask_image", "b"), Data([4, 5])),
            (member("edit_images", "c"), Data([6])),
            (member("edit_images", "d"), Data([7])),
        ], into: emptyRequest)
        #expect(relayed.sourceImage == Data([1, 2, 3]).base64EncodedString())
        #expect(relayed.maskImage == Data([4, 5]).base64EncodedString())
        #expect(relayed.editImages == [Data([6]).base64EncodedString(),
                                       Data([7]).base64EncodedString()])
    }

    /// Keyframes are the one role the host retains as a DOCUMENT rather than
    /// a picture, so they are decoded rather than base64'd.
    @Test func relaysAKeyframeAsTheKeyframeItIs() throws {
        let document = Data(#"{"frame":12,"image":"QUJD","name":"open.png"}"#.utf8)
        let relayed = try RetainedSourceMedia.relayed(
            [(member("keyframes", "a"), document)], into: emptyRequest)
        #expect(relayed.keyframes == [KeyframeCondition(frame: 12, image: "QUJD",
                                                        name: "open.png")])
    }

    /// The failure in the MIDDLE of a set: nothing may be written when a
    /// later member has nowhere to go, or the request goes out half-relayed.
    @Test func aConflictLateInTheSetLeavesTheRequestUntouched() {
        var held = emptyRequest
        held.maskImage = "MINE"
        #expect(throws: RetainedSourceMedia.RelayFailure.alreadyHeld(.maskImage)) {
            try RetainedSourceMedia.relayed([
                (member("source_image", "a"), Data([1])),
                (member("mask_image", "b"), Data([2])),
            ], into: held)
        }
    }

    @Test func aRoleThisBuildCannotPlaceRefusesRatherThanBeingDroppedQuietly() {
        #expect(throws: RetainedSourceMedia.RelayFailure.unsupportedRole("references")) {
            try RetainedSourceMedia.relayed(
                [(member("references", "a"), Data([1]))], into: emptyRequest)
        }
    }

    /// **Fails today**: nothing reads `sizeBytes`, so a relay downloads the
    /// whole member and only then finds the request cannot be sent.
    @Test func refusesARelayThatCouldNeverBeSentBeforeFetchingAByte() throws {
        let big = RetainedSourceMedia.Member(
            memberId: "m", role: "source_video", displayName: "clip",
            sizeBytes: 400 * 1_024 * 1_024)
        // One copy already exceeds what a machine accepts, base64 included.
        let one = try #require(RetainedSourceMedia.relayRefusal([big], copies: 1))
        // The machine's OWN limit, in the one sentence that spells it.
        #expect(one.errorDescription?.contains(RequestBodyLimit.sentence) == true)

        // Four siblings of a member that fits alone do NOT: the body carries
        // it once per sibling.
        let modest = RetainedSourceMedia.Member(
            memberId: "m", role: "source_image", displayName: "picture",
            sizeBytes: 30 * 1_024 * 1_024)
        #expect(RetainedSourceMedia.relayRefusal([modest], copies: 1) == nil)
        let four = try #require(RetainedSourceMedia.relayRefusal([modest], copies: 4))
        #expect(four.errorDescription?.contains("4 copies") == true)

        // An ordinary picture across four siblings is fine.
        let ordinary = RetainedSourceMedia.Member(
            memberId: "m", role: "source_image", displayName: "picture",
            sizeBytes: 2 * 1_024 * 1_024)
        #expect(RetainedSourceMedia.relayRefusal([ordinary], copies: 4) == nil)
    }

    @Test func theRelaySizeIsTheBodyItWouldSendNotTheBytesOnDisk() {
        let member = RetainedSourceMedia.Member(
            memberId: "m", role: "source_image", displayName: "p", sizeBytes: 3_000)
        // base64 is 4 bytes per 3, once per sibling.
        #expect(RetainedSourceMedia.relayBodyBytes([member], copies: 1) == 4_000)
        #expect(RetainedSourceMedia.relayBodyBytes([member], copies: 4) == 16_000)
        #expect(RetainedSourceMedia.relayBodyBytes([], copies: 4) == 0)
    }

    // MARK: - The one-use handle

    @Test func theHandleRidesAHeaderAndNeverTheBody() throws {
        let admission = BatchAdmission(
            clientBatchId: "batch-1",
            requests: [emptyRequest], retainedMediaSession: "secret-handle")
        let body = try MoldJSON.encoder.encode(admission)
        let text = try #require(String(data: body, encoding: .utf8))
        #expect(!text.contains("secret-handle"))
        #expect(text.contains("batch-1"))
        // And a round trip through the wire shape drops it, which is what
        // keeps it out of a persisted draft or a recovery record.
        let read = try MoldJSON.decoder.decode(BatchAdmission.self, from: body)
        #expect(read.retainedMediaSession == nil)
        #expect(read.clientBatchId == "batch-1")
    }

    @Test func aBatchOfMoreThanOneIsRefusedInTheHostsOwnWords() {
        let one = BatchAdmission(requests: [emptyRequest], retainedMediaSession: "h")
        #expect(one.retainedMediaBatchRefusal == nil)
        let four = BatchAdmission(requests: Array(repeating: emptyRequest, count: 4),
                                  retainedMediaSession: "h")
        guard case let .http(status, code, message)? = four.retainedMediaBatchRefusal else {
            Issue.record("a batch of four with a handle must refuse")
            return
        }
        #expect(status == 422)
        #expect(code == "RETAINED_MEDIA_REUSE_BATCH_AMBIGUOUS")
        #expect(message == "a retained-media reuse session binds exactly one batch child")
        // Without a handle, a batch of four is an ordinary batch of four.
        #expect(BatchAdmission(requests: Array(repeating: emptyRequest, count: 4))
            .retainedMediaBatchRefusal == nil)
    }
}
