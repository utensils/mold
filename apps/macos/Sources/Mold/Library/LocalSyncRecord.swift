import CryptoKit
import Foundation
import MoldClient
import Synchronization

/// A durable origin-to-local filename link. A gallery's media version is its
/// server-owned change token; the recipe digest and source identity prevent
/// two hosts with the same basename from sharing a link. The destination is
/// checked against the current local listing before a transfer is skipped.
///
/// A `provisional` record is the output receipt a transfer writes the moment
/// the destination's import answered: the copy's name is proven, its listed
/// facts are not read yet. It never carries checkpoint tokens, so it can only
/// shorten a resume (no second output download), never skip one.
nonisolated struct LocalSyncRecord: Codable, Sendable, Equatable {
    let sourceVersion: String
    let sourceTimestamp: UInt64
    let sourceSize: Int
    let sourceRecipe: String
    let destinationFilename: String
    let destinationVersion: String
    let destinationTimestamp: UInt64
    let destinationSize: Int
    let destinationRecipe: String
    let destinationInstance: String?
    var sourceCheckpoint: String?
    var destinationCheckpoint: String?
    var sourceRoute: String?
    var destinationRoute: String?
    var provisional: Bool?

    static func key(hostID: MoldHost.ID, filename: String) -> String {
        "\(hostID.uuidString):\(filename)"
    }

    /// The source host and filename a key names.
    static func source(of key: String) -> (host: MoldHost.ID, filename: String)? {
        guard let colon = key.firstIndex(of: ":"),
              let host = UUID(uuidString: String(key[..<colon])) else { return nil }
        return (host, String(key[key.index(after: colon)...]))
    }

    /// Whether both checkpoint tokens bind this record: the only state a
    /// cursor-scoped plan may leave unconsidered.
    var sealed: Bool {
        provisional != true && sourceCheckpoint != nil && destinationCheckpoint != nil
            && sourceRoute != nil && destinationRoute != nil
    }

    /// SHA-256 of the canonical recipe, cached per (host, filename, media
    /// version, digest of the exact raw recipe bytes): canonicalizing
    /// re-parses and re-serializes the recipe, which a plan must not repeat
    /// for every print on every run, and the raw digest makes an edited
    /// recipe a different key rather than a stale hit. A row without raw
    /// recipe bytes is never cached.
    static func recipe(_ print: GalleryPrint, host: MoldHost.ID) -> String? {
        let cacheKey = cacheKey(print, host: host)
        if let cacheKey, let cached = recipeCache.withLock({ $0[cacheKey] }) { return cached }
        guard let encoded = print.canonicalMetadataJSON else { return nil }
        let digest = SHA256.hash(data: encoded).map { String(format: "%02x", $0) }.joined()
        if let cacheKey {
            recipeCache.withLock { cache in
                if cache.count >= recipeCacheLimit { cache.removeAll(keepingCapacity: true) }
                cache[cacheKey] = digest
            }
        }
        return digest
    }

    private static let recipeCacheLimit = 200_000
    private static let recipeCache = Mutex<[String: String]>([:])
    /// Whether this print's digest is cached. For tests.
    static func recipeCached(_ print: GalleryPrint, host: MoldHost.ID) -> Bool {
        guard let key = cacheKey(print, host: host) else { return false }
        return recipeCache.withLock { $0[key] != nil }
    }

    private static func cacheKey(_ print: GalleryPrint, host: MoldHost.ID) -> String? {
        guard let raw = print.rawMetadataDigest else { return nil }
        return [host.uuidString, print.filename, print.mediaVersion ?? "-", raw].joined(separator: "\u{1F}")
    }

    func matches(source: GalleryPrint, from sourceHost: MoldHost.ID,
                 local: GalleryPrint, on localHost: MoldHost.ID) -> Bool {
        guard source.mediaVersion == sourceVersion, source.timestamp == sourceTimestamp,
              source.sizeBytes == sourceSize, local.filename == destinationFilename
        else { return false }
        // A receipt names the copy an import answered with, nothing more: it
        // links only through the occupied-name byte comparison.
        guard provisional != true else { return false }
        return local.mediaVersion == destinationVersion
            && local.timestamp == destinationTimestamp
            && local.sizeBytes == destinationSize
            && Self.recipe(source, host: sourceHost) == sourceRecipe
            && Self.recipe(local, host: localHost) == destinationRecipe
    }

    init?(source: GalleryPrint, from sourceHost: MoldHost.ID, local: GalleryPrint, on localHost: MoldHost.ID,
          destinationInstance: String? = nil) {
        guard let sourceVersion = source.mediaVersion,
              let sourceSize = source.sizeBytes,
              let sourceRecipe = Self.recipe(source, host: sourceHost),
              let destinationVersion = local.mediaVersion,
              let destinationSize = local.sizeBytes,
              let destinationRecipe = Self.recipe(local, host: localHost)
        else { return nil }
        self.sourceVersion = sourceVersion
        sourceTimestamp = source.timestamp
        self.sourceSize = sourceSize
        self.sourceRecipe = sourceRecipe
        destinationFilename = local.filename
        self.destinationVersion = destinationVersion
        destinationTimestamp = local.timestamp
        self.destinationSize = destinationSize
        self.destinationRecipe = destinationRecipe
        self.destinationInstance = destinationInstance
    }

    /// The output receipt for an import the destination answered with `filename`.
    init?(receipt source: GalleryPrint, from sourceHost: MoldHost.ID, as filename: String,
          destinationInstance: String?) {
        guard let sourceVersion = source.mediaVersion, let sourceSize = source.sizeBytes,
              let sourceRecipe = Self.recipe(source, host: sourceHost) else { return nil }
        self.sourceVersion = sourceVersion
        sourceTimestamp = source.timestamp
        self.sourceSize = sourceSize
        self.sourceRecipe = sourceRecipe
        destinationFilename = filename
        destinationVersion = sourceVersion
        destinationTimestamp = source.timestamp
        destinationSize = sourceSize
        destinationRecipe = sourceRecipe
        self.destinationInstance = destinationInstance
        provisional = true
    }
}
