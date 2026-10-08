import Foundation

extension FixtureMachine {
    static func frameReuseGallery(_ data: Data) throws -> Data {
        var rows = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]
        rows[0]["metadata"] = ["prompt": "Fixture 0", "model": "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p", "frames": 141,
            "width": 1280, "height": 720, "source_image_name": "opening.png", "source_image_sha256": "opening",
            "keyframes": [["frame": 140, "name": "closing.png", "sha256": "closing"]]]
        return try JSONSerialization.data(withJSONObject: rows)
    }

    func retainedFrameResponse(_ path: String) -> Data? {
        guard retainedFrameFixture else { return nil }
        let prefix = "/api/gallery/source-media/fixture-0.png"
        let image = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")!
        let frame = try! JSONSerialization.data(withJSONObject: ["frame": 140, "image": image.base64EncodedString(), "name": "closing.png"])
        if path == prefix {
            return try! JSONSerialization.data(withJSONObject: ["availability": "available", "members": [
                ["member_id": "opening", "role": "source_image", "display_name": "opening.png", "size_bytes": image.count],
                ["member_id": "closing", "role": "keyframes", "display_name": "closing.png", "size_bytes": frame.count]]])
        }
        if path == prefix + "/opening" { return image }
        if path == prefix + "/closing" { return frame }
        return nil
    }
}
