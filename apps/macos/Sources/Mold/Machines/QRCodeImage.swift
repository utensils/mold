import CoreImage
import CoreImage.CIFilterBuiltins
import SwiftUI

/// A QR code, scaled by a whole-number transform before rasterizing so its
/// modules stay crisp rather than blurred by the view's own interpolation.
/// The only file in the app allowed to import `CoreImage` (`make lint`).
struct QRCodeImage: View {
    let payload: String

    var body: some View {
        if let image = Self.render(payload) {
            Image(decorative: image, scale: 1)
                .interpolation(.none)
                .resizable()
                .aspectRatio(1, contentMode: .fit)
        } else {
            Rectangle()
                .fill(.secondary.opacity(0.15))
                .aspectRatio(1, contentMode: .fit)
        }
    }

    /// Correction level `M`: enough to survive a scuffed phone screen
    /// without the modules getting so fine a camera has to be right on top
    /// of it.
    static func render(_ payload: String, scale: CGFloat = 10) -> CGImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = Data(payload.utf8)
        filter.correctionLevel = "M"
        guard let output = filter.outputImage else { return nil }
        let scaled = output.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return CIContext().createCGImage(scaled, from: scaled.extent)
    }
}
