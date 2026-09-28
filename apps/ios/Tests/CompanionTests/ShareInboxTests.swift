import CoreGraphics
import Darwin
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import MoldCompanion

/// The Share extension's staging: a 48 MP photo becomes 2048 px without
/// ever being decoded whole (an extension dies past ~120 MB), alpha stays
/// PNG, and the app finds, reads and removes what is waiting.
struct ShareInboxTests {
    private func directory() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "inbox-\(UUID())")
    }

    /// A picture written to disk, drawn in grey to keep the test's own peak low.
    private func picture(width: Int, height: Int, alpha: Bool = false, type: UTType = .jpeg) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: "\(UUID()).\(type.preferredFilenameExtension!)")
        let context = alpha
            ? CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            : CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        context.setFillColor(gray: 0.5, alpha: alpha ? 0.5 : 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let out = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(out, context.makeImage()!, nil)
        #expect(CGImageDestinationFinalize(out))
        return url
    }

    private func peakFootprint() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        _ = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return info.ledger_phys_footprint_peak
    }

    @Test func a48MegapixelPhotoIsStagedAt2048WithoutDecodingItWhole() throws {
        let source = try picture(width: 8000, height: 6000)
        let before = peakFootprint()
        let item = try ShareInbox.stage(source, use: .source, in: directory())
        let grew = peakFootprint() - before
        #expect(max(item.width, item.height) == 2048)
        // A whole RGBA decode is 192 MB on its own.
        #expect(grew < 100 << 20, "staging raised the peak by \(grew >> 20) MB")
    }

    @Test func transparencyIsKeptAsPNG() throws {
        let dir = directory()
        let item = try ShareInbox.stage(try picture(width: 300, height: 200, alpha: true, type: .png), use: .reference, in: dir)
        #expect(item.file.hasSuffix(".png"))
        #expect(item.width == 300)
    }

    @Test func whatIsWaitingIsListedOldestFirstAndRemoved() throws {
        let dir = directory()
        let first = try ShareInbox.stage(try picture(width: 64, height: 64), use: .library, in: dir,
                                         now: Date(timeIntervalSince1970: 1))
        let second = try ShareInbox.stage(try picture(width: 64, height: 64), use: .source, in: dir,
                                          now: Date(timeIntervalSince1970: 2))
        #expect(ShareInbox.items(in: dir).map(\.id) == [first.id, second.id])
        ShareInbox.remove(first, in: dir)
        #expect(ShareInbox.items(in: dir).map(\.id) == [second.id])
    }

    @Test func somethingThatIsNotAPictureIsRefused() {
        #expect(throws: ShareInbox.Failure.unreadable) {
            try ShareInbox.stage(data: Data("not a picture".utf8), use: .source, in: directory())
        }
    }
}
