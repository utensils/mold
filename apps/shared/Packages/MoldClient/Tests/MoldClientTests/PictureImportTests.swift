import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import MoldClient

/// A picture the machine cannot read is converted HERE -- HEIC, the format
/// every iPhone photo is -- and one it can read is sent as the exact bytes.
struct PictureImportTests {
    private func encoded(as type: UTType) throws -> Data {
        let context = try #require(CGContext(data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 32,
                                             space: CGColorSpaceCreateDeviceRGB(),
                                             bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        let image = try #require(context.makeImage())
        let data = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(data, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return data as Data
    }

    @Test func aReadableFormatIsSentAsTheExactBytes() throws {
        let png = try encoded(as: .png)
        let picked = try PictureImport.conform(png, name: "a.png", accepting: PictureImport.engineReadable)
        #expect(picked.data == png)
        #expect(picked.name == "a.png")
    }

    @Test(.enabled(if: ((CGImageDestinationCopyTypeIdentifiers() as? [String]) ?? []).contains(UTType.heic.identifier)))
    func aHEICPhotoBecomesAPNG() throws {
        let heic = try encoded(as: .heic)
        let picked = try PictureImport.conform(heic, name: "IMG_0001.HEIC", accepting: PictureImport.engineReadable)
        #expect(picked.name == "IMG_0001.png")
        #expect(picked.data.starts(with: [0x89, 0x50, 0x4E, 0x47]))
        #expect(PictureImport.pixelSize(of: picked.data)?.width == 8)
    }

    @Test func somethingThatIsNotAPictureIsRefusedByName() {
        #expect(throws: PictureImportError.self) {
            try PictureImport.conform(Data("hello".utf8), name: "notes.txt", accepting: PictureImport.engineReadable)
        }
    }
}
