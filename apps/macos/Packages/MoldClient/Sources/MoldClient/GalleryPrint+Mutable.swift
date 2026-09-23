import Foundation

public extension GalleryPrint {
    /// A print with the few fields a client may change locally.
    ///
    /// `GalleryPrint` is a wire type and stays immutable; this exists so an
    /// optimistic update can turn a star on without inventing a second model
    /// of what a print is.
    struct Mutable {
        public var favorite: Bool?
        public var tags: [String]?
        public var title: String?
        /// This machine's own ids for the shelves the print sits on. Filing is
        /// a local change like any other -- the grid draws a collection scope
        /// from these, so a print filed optimistically has to leave the shelf
        /// it was dragged off immediately.
        public var collections: [String]?
        private let base: GalleryPrint

        public init(_ print: GalleryPrint) {
            self.base = print
            self.favorite = print.favorite
            self.tags = print.tags
            self.title = print.title
            self.collections = print.collections
        }

        public func build() -> GalleryPrint {
            GalleryPrint(
                filename: base.filename, metadata: base.metadata, timestamp: base.timestamp,
                format: base.format, sizeBytes: base.sizeBytes, mediaVersion: base.mediaVersion,
                title: title, tags: tags, favorite: favorite, collections: collections,
                trashedAt: base.trashedAt, purgeAt: base.purgeAt,
                metadataSynthetic: base.metadataSynthetic,
                rawMetadataJSON: base.rawMetadataJSON)
        }
    }
}
