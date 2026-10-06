import SwiftUI
import UIKit
import MoldClient

/// Native menus flatten their label into a UIImage: draw the actual ratio
/// there rather than handing every aspect the same SF Symbol rectangle.
enum AspectRatioIcon {
    static func size(width: Int, height: Int, bound: CGFloat) -> CGSize {
        AspectRatioGeometry.size(width: width, height: height, bound: bound)
    }

    static func image(width: Int, height: Int) -> UIImage {
        let bounds = CGSize(width: 26, height: 26)
        let size = size(width: width, height: height, bound: 22)
        return UIGraphicsImageRenderer(size: bounds).image { _ in
            let rect = CGRect(x: (bounds.width - size.width) / 2,
                              y: (bounds.height - size.height) / 2,
                              width: size.width, height: size.height)
            let outline = UIBezierPath(roundedRect: rect, cornerRadius: 2)
            outline.lineWidth = 1.5
            UIColor.label.setStroke()
            outline.stroke()
        }.withRenderingMode(.alwaysTemplate)
    }
}
