import Foundation

public nonisolated enum AspectRatioGeometry {
    public static func size(width: Int, height: Int, bound: CGFloat) -> CGSize {
        let longest = CGFloat(max(1, max(width, height)))
        return CGSize(width: CGFloat(max(1, width)) / longest * bound,
            height: CGFloat(max(1, height)) / longest * bound)
    }
}
