import Foundation

public struct ExpandReference: Codable, Hashable, Sendable {
    public var kind: String
    public var hasAudio: Bool
    public var role: String
}
/// Payload-free conditioning context; no filenames, digests or media authorities.
public struct ExpandContext: Codable, Hashable, Sendable {
    public var width: Int
    public var height: Int
    public var frames: Int?
    public var fps: Int?
    public var audio: Bool?
    public var references: [ExpandReference]
    public init(request: GenerateRequest) {
        width = request.width; height = request.height
        frames = request.frames; fps = request.fps; audio = request.enableAudio
        references = (request.references ?? []).map {
            ExpandReference(kind: $0.kind == "named_image" ? "image" : $0.kind,
                hasAudio: $0.hasAudio == true, role: "reference")
        }
        references += (request.editImages ?? []).map { _ in
            ExpandReference(kind: "image", hasAudio: false, role: "edit")
        }
        if request.sourceImage != nil {
            references.append(.init(kind: "image", hasAudio: false, role: frames == nil ? "source" : "first-frame"))
        }
        references += (request.keyframes ?? []).map { _ in
            ExpandReference(kind: "image", hasAudio: false, role: "keyframe")
        }
    }
}
