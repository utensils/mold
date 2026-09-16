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
        private let base: GalleryPrint

        public init(_ print: GalleryPrint) {
            self.base = print
            self.favorite = print.favorite
            self.tags = print.tags
            self.title = print.title
        }

        public func build() -> GalleryPrint {
            GalleryPrint(
                filename: base.filename, metadata: base.metadata, timestamp: base.timestamp,
                format: base.format, sizeBytes: base.sizeBytes, mediaVersion: base.mediaVersion,
                title: title, tags: tags, favorite: favorite, collections: base.collections,
                trashedAt: base.trashedAt, purgeAt: base.purgeAt)
        }
    }
}
