import Foundation

/// Presentation-only classification. It never changes the source's member identity.
public enum SourceMediaFilter: String, CaseIterable, Identifiable, Sendable {
    case all, image, video, audio, unknown
    public var id: String { rawValue }
    public var title: String {
        switch self { case .all: "All media"; case .image: "Images"; case .video: "Videos"; case .audio: "Audio"; case .unknown: "Other" }
    }
    public func includes(_ member: RetainedSourceMedia.Member) -> Bool {
        self == .all || rawValue == member.contentKind
    }
}
