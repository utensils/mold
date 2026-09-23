import Foundation

/// A file on its way into a machine's library.
///
/// The body is a length-prefixed frame rather than multipart, because that is
/// what lets a host start writing a 60 GB file before it has read all of it:
/// four bytes of descriptor length, eight of file length, the descriptor, then
/// the bytes.
public struct GalleryImport: Sendable {
    public var prompt: String
    public var model: String
    /// The picture's real dimensions where they are knowable, and zero where
    /// they are not. The field is not optional on the wire.
    public var width: Int
    public var height: Int
    /// Which mold wrote this row. For an import that is the machine
    /// receiving it, because that is what is creating the row.
    public var version: String
    public var file: Data
    /// When it was made. Absent means now, which puts an old picture at the
    /// top of today rather than where it belongs.
    public var timestamp: Date?
    /// An existing Mold print carries its own recipe; Finder imports do not.
    public var originalMetadata: OutputMetadata?
    var originalMetadataJSON: Data?
    public var metadataSynthetic = true

    public init(prompt: String, model: String, width: Int = 0, height: Int = 0,
                version: String, file: Data, timestamp: Date? = nil) {
        self.prompt = prompt
        self.model = model
        self.width = width
        self.height = height
        self.version = version
        self.file = file
        self.timestamp = timestamp
        self.originalMetadata = nil
        self.originalMetadataJSON = nil
    }

    /// An arbitrary file from this Mac.
    ///
    /// `prompt` is not optional on the wire and an imported picture has none,
    /// so it is described by where it came from. `model` says `import` for the
    /// same reason: the field has to say something, and a fiction naming a
    /// real checkpoint would make the print look reusable when it is not.
    public init(importing file: Data, named name: String, version: String,
                width: Int = 0, height: Int = 0, madeAt: Date? = nil) {
        self.init(prompt: "Imported \u{2014} \(name)", model: "import",
                  width: width, height: height, version: version,
                  file: file, timestamp: madeAt)
    }

    public static let contentType = "application/vnd.mold.gallery-import"

    /// Copy an existing print into another host's Library without changing
    /// the bytes, recipe, or when the print was made.
    public init(mirroring print: GalleryPrint, file: Data) {
        self.init(prompt: print.metadata.prompt ?? "", model: print.metadata.model ?? "",
                  version: print.metadata.version ?? "", file: file,
                  timestamp: print.createdAt)
        originalMetadata = print.metadata
        originalMetadataJSON = print.rawMetadataJSON
        metadataSynthetic = print.metadataSynthetic ?? false
    }

    public func body() throws -> Data {
        let metadata: [String: Any]
        if let originalMetadataJSON {
            guard let object = try JSONSerialization.jsonObject(with: originalMetadataJSON) as? [String: Any] else {
                throw MoldClientError.malformedResponse
            }
            metadata = object
        } else if let originalMetadata {
            let encoded = try MoldJSON.encoder.encode(originalMetadata)
            guard let object = try JSONSerialization.jsonObject(with: encoded) as? [String: Any] else {
                throw MoldClientError.malformedResponse
            }
            var normalized = object
            // Foundation's acronym conversion writes `sha256_s` for these
            // two fields, while Rust's wire keys end in `sha256s`.
            for prefix in ["edit_image", "id_image"] {
                let generated = "\(prefix)_sha256_s"
                if let digests = normalized.removeValue(forKey: generated) {
                    normalized["\(prefix)_sha256s"] = digests
                }
            }
            metadata = normalized
        } else {
            metadata = [
                "prompt": prompt, "model": model, "seed": 0, "steps": 0,
                "guidance": 0, "width": width, "height": height, "version": version,
            ]
        }
        var descriptor: [String: Any] = [
            // Eight of these are NOT optional on the wire, and a descriptor
            // missing any one is a 422 naming only the first. A render has
            // real values; an import has none, and zero is the honest answer
            // rather than a plausible-looking fiction somebody might reuse.
            "metadata": metadata,
            // Finder imports invented metadata; mirrored prints carry their
            // source host's synthetic flag instead.
            "metadata_synthetic": metadataSynthetic,
        ]
        if let timestamp {
            descriptor["timestamp"] = UInt64(timestamp.timeIntervalSince1970)
        }
        let json = try JSONSerialization.data(withJSONObject: descriptor,
                                               options: [.sortedKeys])

        var body = Data(capacity: 12 + json.count + file.count)
        // Big endian, and the widths are not interchangeable: four for the
        // descriptor, eight for the file.
        withUnsafeBytes(of: UInt32(json.count).bigEndian) { body.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt64(file.count).bigEndian) { body.append(contentsOf: $0) }
        body.append(json)
        body.append(file)
        return body
    }
}

extension HTTPBackend {
    /// Puts a file into this machine's library.
    ///
    /// Idempotent by content: the same bytes with the same metadata under the
    /// same name keep that name rather than minting `holiday-2.png`, which is
    /// what makes a retry safe and a re-import harmless.
    @discardableResult
    public func importPrint(_ item: GalleryImport, as filename: String) async throws -> String {
        var request = self.request("/api/gallery/import/\(escaped(filename))")
        request.httpMethod = "PUT"
        request.setValue(GalleryImport.contentType, forHTTPHeaderField: "Content-Type")
        request.httpBody = try item.body()
        let data = try await bytes(for: request)
        struct Answer: Decodable { let filename: String }
        guard let answer = try? MoldJSON.decoder.decode(Answer.self, from: data) else {
            throw MoldClientError.malformedResponse
        }
        return answer.filename
    }
}
