import Foundation
import MoldClient

/// Two-line fixtures: this bundle sees only MoldClient's PUBLIC surface, and
/// neither `GalleryPrint`'s memberwise init nor `QueueEntry`'s is public, so a
/// print or a queue row is built the way the wire builds one -- by decoding it.
enum FakeFixtures {
    static func print(_ filename: String, prompt: String? = nil) -> GalleryPrint {
        let json = """
        {"filename": "\(filename)", "metadata": {"prompt": \(prompt.map { "\"\($0)\"" } ?? "null")},
         "timestamp": 1000}
        """
        return try! MoldJSON.decoder.decode(GalleryPrint.self, from: Data(json.utf8))
    }

    static func queueEntry(_ id: String, state: String = "accepted") -> QueueEntry {
        let json = #"{"id": "\#(id)", "state": "\#(state)"}"#
        return try! MoldJSON.decoder.decode(QueueEntry.self, from: Data(json.utf8))
    }
}
