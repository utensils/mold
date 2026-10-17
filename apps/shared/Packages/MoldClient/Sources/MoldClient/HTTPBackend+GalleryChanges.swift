import Foundation

// `GET /api/gallery/changes`: the change log a `GalleryReplica` follows.
public extension HTTPBackend {
    /// The server caps a page at this many changes.
    static let galleryChangesPageLimit = 2_000

    func galleryChanges(since cursor: String?, limit: Int) async throws -> GalleryChangesPage {
        let bounded = min(max(limit, 1), Self.galleryChangesPageLimit)
        var path = "/api/gallery/changes?limit=\(RouteEscaping.escapedQueryValue(String(bounded)))"
        if let cursor { path += "&since=" + RouteEscaping.escapedQueryValue(cursor) }
        var request = self.request(path)
        // A snapshot page of 2,000 rows is a listing-sized answer.
        request.timeoutInterval = 60
        let (data, http) = try await send(request)
        try HTTPRefusal.check(http, data)
        let page: GalleryChangesPage
        do {
            page = try MoldJSON.decoder.decode(GalleryChangesPage.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
        guard page.protocolVersion == 1 else { throw MoldClientError.malformedResponse }
        return Self.preservingRawMetadata(page, from: data)
    }

    /// The listing keeps each row's original recipe JSON for a byte-preserving
    /// mirror (`GalleryPrint.rawMetadataJSON`); a feed row must carry it too,
    /// or a print first seen through the feed would import a lossier recipe.
    internal static func preservingRawMetadata(_ page: GalleryChangesPage, from data: Data) -> GalleryChangesPage {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = object["changes"] as? [[String: Any]] else { return page }
        var raw: [String: Data] = [:]
        for row in rows {
            guard let image = row["image"] as? [String: Any],
                  let name = image["filename"] as? String,
                  let metadata = image["metadata"] as? [String: Any],
                  let encoded = try? JSONSerialization.data(withJSONObject: metadata, options: .sortedKeys) else { continue }
            raw[name] = encoded
        }
        let changes = page.changes.map { change -> GalleryChange in
            guard var print = change.image else { return change }
            print.rawMetadataJSON = raw[print.filename]
            return GalleryChange(filename: change.filename, image: print)
        }
        return GalleryChangesPage(protocolVersion: page.protocolVersion, instanceId: page.instanceId,
                                  cursor: page.cursor, more: page.more, reset: page.reset,
                                  changes: changes)
    }
}
