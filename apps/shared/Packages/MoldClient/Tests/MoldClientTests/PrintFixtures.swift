import Foundation

@testable import MoldClient

/// Building a print for a test.
///
/// `GalleryPrint` is a wire type with sixteen fields, most of them optional
/// because they span years of mold versions. Spelling all of them out at every
/// call site buries the one field a test is actually about.
enum PrintFixtures {
    static func metadata(prompt: String? = nil, model: String? = nil,
                         seed: UInt64? = nil, frames: Int? = nil) -> OutputMetadata {
        var json: [String: Any] = [:]
        if let prompt { json["prompt"] = prompt }
        if let model { json["model"] = model }
        if let seed { json["seed"] = seed }
        if let frames { json["frames"] = frames }
        let data = try! JSONSerialization.data(withJSONObject: json)
        return try! MoldJSON.decoder.decode(OutputMetadata.self, from: data)
    }

    static func print(_ filename: String, timestamp: UInt64 = 1_000, format: String? = "png",
                     tags: [String]? = nil, favorite: Bool? = nil, title: String? = nil,
                     collections: [String]? = nil, prompt: String? = nil,
                     bytes: Int? = 100, trashedAt: UInt64? = nil,
                     purgeAt: UInt64? = nil, frames: Int? = nil) -> GalleryPrint {
        GalleryPrint(
            filename: filename, metadata: metadata(prompt: prompt, frames: frames),
            timestamp: timestamp,
            format: format, sizeBytes: bytes, mediaVersion: "v1", title: title, tags: tags,
            favorite: favorite, collections: collections, trashedAt: trashedAt, purgeAt: purgeAt)
    }

    static func entry(_ filename: String, host: UUID, hostName: String = "workstation",
                      timestamp: UInt64 = 1_000, format: String = "png", tags: [String]? = nil,
                      favorite: Bool? = nil, title: String? = nil, collections: [String]? = nil,
                      prompt: String? = nil, bytes: Int? = 100) -> LibraryEntry {
        LibraryEntry(
            host: MoldHost(id: host, name: hostName, baseURL: URL(string: "http://h")!),
            print: print(filename, timestamp: timestamp, format: format, tags: tags,
                         favorite: favorite, title: title, collections: collections,
                         prompt: prompt, bytes: bytes)
        )
    }
}
