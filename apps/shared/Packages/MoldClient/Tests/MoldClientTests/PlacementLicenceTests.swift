import Foundation
import Testing

@testable import MoldClient

// `GenerationPlacementPreview.pending_downloads` is a list of OBJECTS
// (`PendingModelDownload`, `types.rs`), and `missing_components` a list of
// `ModelComponentStatus`. Both were typed `[String]` here, so any preview for
// an uninstalled model -- Qwen Image 2.1 before its licence is accepted, the
// case that matters -- failed to decode at all.

private let uninstalledQwen21 = """
{"version":1,"authoritative":true,"state_version":4,"plan_version":2,
 "outcome":"infeasible","reason":"qwen-image-2.1:bf16 is not installed.",
 "pending_downloads":[
   {"kind":"model","name":"qwen-image-2.1:bf16","repo":"Qwen/Qwen-Image-2.1",
    "bytes":41000000000,"install_model":"qwen-image-2.1:bf16",
    "licenses":[{"id":"qwen-research","name":"Qwen Research License",
      "url":"https://huggingface.co/Qwen/Qwen-Image-2.1/raw/abc/LICENSE",
      "canonical":"https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
      "sha256":"0123456789abcdef","summary":"Research and non-commercial use."}]},
   {"kind":"encoder","name":"qwen2.5-vl","repo":"Qwen/Qwen2.5-VL","bytes":16000000000,
    "licenses":[{"id":"qwen-research","name":"Qwen Research License",
      "url":"https://huggingface.co/Qwen/Qwen-Image-2.1/raw/abc/LICENSE",
      "canonical":"https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
      "sha256":"0123456789abcdef","summary":"Research and non-commercial use."}]}
 ],
 "missing_components":[{"kind":"transformer","name":"qwen-image-2.1","present":false}]}
"""

@Test func aPreviewForAnUninstalledGatedModelDecodes() throws {
    let preview = try MoldJSON.decoder.decode(
        PlacementPreview.self, from: Data(uninstalledQwen21.utf8))
    #expect(preview.pendingDownloads?.count == 2)
    #expect(preview.pendingDownloads?.first?.installModel == "qwen-image-2.1:bf16")
    #expect(preview.missingComponents?.first?.name == "qwen-image-2.1")
}

@Test func theLicencesToAskForAreEachTermOnceInOrder() throws {
    let preview = try MoldJSON.decoder.decode(
        PlacementPreview.self, from: Data(uninstalledQwen21.utf8))
    #expect(preview.outstandingLicenses.map(\.id) == ["qwen-research"])
    #expect(preview.outstandingLicenses(excluding: ["qwen-research"]).isEmpty)
}

@Test func aPreviewWithNothingToDownloadAsksForNothing() throws {
    let json = #"{"outcome":"planned"}"#
    let preview = try MoldJSON.decoder.decode(PlacementPreview.self, from: Data(json.utf8))
    #expect(preview.outstandingLicenses.isEmpty)
}
