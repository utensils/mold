import Foundation
import MoldClient

// The shape logic is `CanvasShape` (MoldClient), shared with the iPhone's
// Shape chip; this keeps the Mac control's own names over it.
extension ShapeControl {
    struct Shape: Equatable {
        let aspect: String
        let aspects: [AspectGroup]
        let sizes: [SizePreset]
        let isOnLadder: Bool
    }

    enum Presentation: Equatable {
        case menus(Shape)
        case fixed(String)
        case fromSource
        case hidden
    }

    static func resolve(resolution: ResolutionProfile, width: Int, height: Int) -> Presentation {
        switch CanvasShape.resolve(resolution, width: width, height: height) {
        case let .menus(menus):
            .menus(Shape(aspect: menus.aspect, aspects: menus.aspects, sizes: menus.sizes,
                         isOnLadder: menus.isOnLadder))
        case let .fixed(text): .fixed(text)
        case .fromSource: .fromSource
        case .hidden: .hidden
        }
    }

    static func size(in group: AspectGroup, nearWidth: Int, height: Int) -> SizePreset? {
        CanvasShape.size(in: group, near: nearWidth, height)
    }
}
