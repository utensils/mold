import Foundation

/// Parameters for `GET /api/catalog/search` (`catalog_api.rs:819-833`).
public struct CatalogQuery: Hashable, Sendable {
    public var text: String?
    public var family: String?
    public var kind: String?
    public var source: String?
    public var sort: String?
    public var page: Int?
    public var pageSize: Int?
    public var includeNSFW: Bool?

    public init(
        text: String? = nil, family: String? = nil, kind: String? = nil,
        source: String? = nil, sort: String? = nil, page: Int? = nil,
        pageSize: Int? = nil, includeNSFW: Bool? = nil
    ) {
        self.text = text
        self.family = family
        self.kind = kind
        self.source = source
        self.sort = sort
        self.page = page
        self.pageSize = pageSize
        self.includeNSFW = includeNSFW
    }

    /// `GET /api/catalog/search`'s query string, omitting anything not
    /// asked for -- an absent field is "no filter" on the server, which is
    /// not the same request as an explicit default. `text` is free user
    /// input; the rest are server-defined tokens, escaped by the same rule
    /// because the rule is about the encoder, not about who supplies the
    /// value. `escapedQueryValue`, never `escaped` -- see its doc comment.
    public var queryString: String {
        var parts: [String] = []
        let escape = RouteEscaping.escapedQueryValue
        if let text, !text.isEmpty { parts.append("q=\(escape(text))") }
        if let family { parts.append("family=\(escape(family))") }
        if let kind { parts.append("kind=\(escape(kind))") }
        if let source { parts.append("source=\(escape(source))") }
        if let sort { parts.append("sort=\(escape(sort))") }
        if let page { parts.append("page=\(page)") }
        if let pageSize { parts.append("page_size=\(pageSize)") }
        if let includeNSFW { parts.append("include_nsfw=\(includeNSFW)") }
        return parts.joined(separator: "&")
    }
}

/// A page of `GET /api/catalog/search` results.
public struct CatalogListing: Codable, Hashable, Sendable {
    public let entries: [CatalogEntry]
    public let page: Int
    public let pageSize: Int
    public let total: Int
    public let providerErrors: [CatalogProviderError]
}

/// One provider's failure inside an otherwise-successful merged search
/// (`live.rs`'s `CatalogProviderError`) -- the other provider's rows still
/// came back, so this rides beside them rather than failing the request.
public struct CatalogProviderError: Codable, Hashable, Sendable {
    public let source: String
    public let message: String
    public let code: String?
    public let retryAfterSeconds: Int?
}

/// One catalog row, decoded from `live_entry_to_wire`
/// (`catalog_api.rs:1238-1325`).
public struct CatalogEntry: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let source: String
    public let sourceId: String
    public let name: String
    public let author: String?
    public let family: String
    public let kind: String
    public let modality: String
    public let sizeBytes: Int64?
    public let downloadCount: Int64
    public let rating: Double?
    public let likes: Int64
    public let nsfw: Bool
    public let thumbnailUrl: String?
    public let description: String?
    public let license: String?
    public let licenseFlags: CatalogLicenseFlags
    public let tags: [String]
    public let companions: [String]
    public let companionDetails: [CatalogCompanionDetail]
    public let supported: Bool
    public let installed: Bool
    public let pageUrl: String?
    public let trainedWords: [String]
}

/// One resolved companion from `entry.companions`, joined against
/// `mold_catalog::companions::COMPANIONS` on the server.
public struct CatalogCompanionDetail: Codable, Hashable, Sendable {
    public let name: String
    public let kind: String
    public let repo: String?
    public let sizeBytes: Int64?
}

/// `LicenseFlags` (`entry.rs:110-114`), all three tri-state.
public struct CatalogLicenseFlags: Codable, Hashable, Sendable {
    public let commercial: Bool?
    public let derivatives: Bool?
    public let differentLicense: Bool?

    /// Whether this row says anything about its licence at all.
    ///
    /// Measured on plato: every row of a three-row search had `license: null`
    /// and all three flags null. All-null is the ORDINARY case and means NO
    /// INFORMATION -- rendering it as "commercial: no" would be a refusal
    /// nobody made (design fact 18/decision 18, M5).
    public var isEmpty: Bool { commercial == nil && derivatives == nil && differentLicense == nil }
}

/// Answer to `POST /api/catalog/:id/download` (`catalog_api.rs:531-561`).
public struct CatalogInstall: Codable, Hashable, Sendable {
    public let primaryJobId: String?
    public let companionJobs: [CompanionJob]

    /// Both ids folded into one list. A null `primaryJobId` with non-empty
    /// `companionJobs` is the "only companions were missing" answer
    /// (`catalog_api.rs:545-561`), not a failure -- so this is never empty
    /// on a real 202.
    public var jobIDs: [String] { [primaryJobId].compactMap { $0 } + companionJobs.map(\.jobId) }
}

/// One queued companion download (`catalog_api.rs:436-444`).
public struct CompanionJob: Codable, Hashable, Sendable {
    public let name: String
    public let jobId: String
}
