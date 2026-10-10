import Foundation

public extension RetainedSourceMedia.Member {
    /// Old manifests have roles, not MIME metadata. Never trust the filename.
    var contentKind: String {
        if let mediaKind { return ["image", "audio", "video"].contains(mediaKind) ? mediaKind : "unknown" }
        if ["audio_file", "audio_file_path"].contains(role) { return "audio" }
        if ["source_video", "source_video_path", "extend_video", "extend_video_path"].contains(role) { return "video" }
        if role.hasPrefix("stage_source:") || ["source_image", "identity_image", "identity_images", "edit_images", "mask_image", "control_image", "keyframes", "matting_processed_source_image", "matting_processed_references"].contains(role) { return "image" }
        return "unknown"
    }
    var sourceLabel: String {
        if let label, !label.isEmpty { return label }
        let name: String
        switch role {
        case "source_image": name = "Source image"
        case "keyframes": name = frameIndex.map { "Frame \($0)" } ?? "Keyframe"
        case "references": name = "Reference"
        case "identity_image", "identity_images": name = "Identity reference"
        case "edit_images": name = "Edit reference"
        case "mask_image": name = "Mask"
        case "control_image": name = "Control image"
        case "audio_file", "audio_file_path": name = "Audio"
        case "source_video", "source_video_path": name = "Source video"
        case "extend_video", "extend_video_path": name = "Continuation video"
        case "matting_processed_source_image", "matting_processed_references": name = "Processed image"
        default: name = role.replacingOccurrences(of: "_", with: " ").capitalized
        }
        return position.map { "\(name) · \($0 + 1)" } ?? name
    }
    var sourceSymbol: String {
        switch contentKind { case "image": "photo"; case "video": "video"; case "audio": "waveform"; default: "doc" }
    }
}

public extension RetainedSourceMedia {
    struct Inventories: Decodable, Sendable {
        public let instanceId: String?
        public let inventories: [String: Inventory]
    }
}
