import Foundation
import Testing
@testable import MoldClient

@Test func videoPauseIsCapabilityGatedAndParkedForOtherFormats() throws {
    let caps = try MoldJSON.decoder.decode(ExportOptions.self, from: Data(#"{"formats":["gif","apng"],"gif_playback":["loop","bounce"],"gif_repeat":["forever","once"],"gif_pause":{"min":0,"max":5000,"step":10,"default":0}}"#.utf8))
    #expect(caps.gifPause?.valid == true)
    let request = VideoExportRequest(format: "gif", playback: .loop, repeatMode: .forever, pauseMs: 0)
    let json = try JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(request)) as! [String: Any]
    #expect(json["pause_ms"] as? Int == 0)
    #expect(json["transparent"] == nil)
    #expect(json["frames"] == nil)
    #expect(VideoExportRequest(format: "apng", pauseMs: 250).effectivePauseMs == nil)
    #expect(VideoExportRequest(format: "gif", repeatMode: .once, pauseMs: 250).effectivePauseMs == nil)
    #expect(VideoExportRequest(format: "gif", playback: .bounce, repeatMode: .once, pauseMs: 250).effectivePauseMs == 250)
    #expect(GifPauseControl(min: 0, max: 5000, step: 0, defaultValue: 0).valid == false)
}

@Test func exportAvailabilityMatchesServerSourceContract() {
    #expect(MediaExportKind(filename: "loop.mp4", trashed: false) == .video)
    #expect(MediaExportKind(filename: "object.glb", trashed: false) == .mesh)
    #expect(MediaExportKind(filename: "loop.gif", trashed: false) == nil)
    #expect(MediaExportKind(filename: "loop.mp4", trashed: true) == nil)
}

@Test func malformedOptionalPauseDoesNotHideOtherExports() throws {
    for pause in [#"{"min":0,"max":5000,"step":"ten","default":0}"#, #"{"min":0,"max":5000,"step":10}"#] {
        let data = Data("{\"formats\":[\"gif\",\"apng\"],\"gif_pause\":\(pause)}".utf8)
        let caps = try MoldJSON.decoder.decode(ExportOptions.self, from: data)
        #expect(caps.forVideo == ["gif", "apng"])
        #expect(caps.gifPause == nil)
    }
}

@Test func assetMetadataSurvivesLocalGalleryRoundTrip() throws {
    let data = Data(#"{"filename":"mesh.glb","metadata":{},"timestamp":1790000000,"assets":[{"asset_id":"base-color","role":"base_color","display_name":"base-color.png","media_type":"image/png","size_bytes":123,"sha256":"abc"}]}"#.utf8)
    let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: data)
    #expect(print.assets?.first?.assetId == "base-color")
    let cached = try MoldJSON.localDecoder.decode(GalleryPrint.self, from: MoldJSON.localEncoder.encode(print))
    #expect(cached.assets == print.assets)
}
