import CoreGraphics
import Foundation
import ImageIO

extension FixtureMachine {
    static func fixtureFile(_ name: String) -> Data {
        let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appending(path: "Fixtures")
        return try! Data(contentsOf: directory.appending(path: name))
    }
    func exportResponse(_ request: [String: Any]) -> Data {
        let format = request["format"] as? String ?? "gif"
        switch format {
        case "zip": return Self.fixtureFile("export-object.zip")
        case "obj": return Data("# fixture\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n".utf8)
        case "ply": return Data("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n".utf8)
        case "stl": return Data("solid fixture\nendsolid fixture\n".utf8)
        default: return Self.exportImage(format: format == "apng" ? "png" : "gif")
        }
    }
    static func exportImage(format: String) -> Data {
        let context = CGContext(data: nil, width: 16, height: 16, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let data = NSMutableData()
        let count = format == "gif" ? 3 : 1
        let destination = CGImageDestinationCreateWithData(data, (format == "gif" ? "com.compuserve.gif" : "public.png") as CFString, count, nil)!
        if format == "gif" { CGImageDestinationSetProperties(destination, [kCGImagePropertyGIFDictionary: [kCGImagePropertyGIFLoopCount: 0]] as CFDictionary) }
        for index in 0..<count {
            context.setFillColor(CGColor(red: CGFloat(index)/3, green: 0.5, blue: 0.5, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: 16, height: 16))
            CGImageDestinationAddImage(destination, context.makeImage()!, [kCGImagePropertyGIFDictionary: [kCGImagePropertyGIFDelayTime: 0.1]] as CFDictionary)
        }
        precondition(CGImageDestinationFinalize(destination))
        return data as Data
    }
}
