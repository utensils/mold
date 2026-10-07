import Foundation
import MoldClient
import Testing

@testable import Mold

@MainActor
struct GifExportSelectionTests {
    private func options() throws -> ExportOptions {
        try MoldJSON.decoder.decode(ExportOptions.self, from: Data(#"{"formats":["gif","apng"],"gif_playback":["loop","bounce"],"gif_repeat":["forever","once"],"gif_pause":{"min":0,"max":5000,"step":10,"default":0}}"#.utf8))
    }

    @Test func zeroPauseIsPreservedAndInactivePauseIsOmitted() throws {
        var selection = GifExportSelection(options: try options())
        #expect(selection.pause(format: "gif", options: try options()) == 0)
        selection.repeatMode = .once
        selection.pauseText = "250"
        #expect(selection.pause(format: "gif", options: try options()) == nil)
        selection.playback = .bounce
        #expect(selection.pause(format: "gif", options: try options()) == 250)
        #expect(selection.pause(format: "apng", options: try options()) == nil)
    }

    @Test func invalidPauseOnlyBlocksAnActiveControl() throws {
        var selection = GifExportSelection(options: try options())
        selection.pauseText = "21"
        #expect(!selection.valid(format: "gif", options: try options()))
        selection.repeatMode = .once
        #expect(selection.valid(format: "gif", options: try options()))
        #expect(selection.valid(format: "apng", options: try options()))
    }

    @Test func olderHostsDoNotReceiveUnadvertisedPause() {
        var selection = GifExportSelection(options: nil)
        selection.pauseText = "200"
        #expect(selection.pause(format: "gif", options: nil) == nil)
        #expect(selection.valid(format: "gif", options: nil))
    }

    @Test func selectionSerializesVideoWithoutMeshKeys() throws {
        let advertised = try options()
        let selection = GifExportSelection(options: advertised)
        let request = VideoExportRequest(playback: selection.playback, repeatMode: selection.repeatMode,
                                         pauseMs: selection.pause(format: "gif", options: advertised))
        let body = try #require(JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(request)) as? [String: Any])
        #expect(body["pause_ms"] as? Int == 0)
        #expect(body["transparent"] == nil)
        #expect(body["frames"] == nil)
        #expect(VideoExportRequest.filename("clip.mp4", format: "apng") == "clip.png")
    }
}
