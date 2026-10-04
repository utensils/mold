import SwiftUI
import Testing
import UIKit

@testable import MoldCompanion

@MainActor
struct BadgeLayoutTests {
    @Test func badgesFitTheProposedTileWidthAtEveryTextSize() throws {
        let sizes: [(DynamicTypeSize, UIContentSizeCategory)] = [
            (.xSmall, .extraSmall), (.large, .large), (.accessibility5, .accessibilityExtraExtraExtraLarge),
        ]
        let symbols: [String?] = [nil, "desktopcomputer"]
        for (size, category) in sizes {
            for width in [CGFloat(80), CGFloat(240)] {
                for text in ["Studio", "Reference media workstation East, Repeated machine, Other media workstation"] {
                    for glyph in symbols {
                        let renderer = ImageRenderer(content: Badge(symbol: glyph, text: text)
                            .environment(\.dynamicTypeSize, size))
                        renderer.proposedSize = ProposedViewSize(width: width, height: nil)
                        renderer.scale = 1
                        let image = try #require(renderer.uiImage)
                        #expect(image.size.width <= width,
                                "A badge must fit the tile proposal at \(size): \(image.size), width \(width)")
                        if size == .accessibility5, width == 80, glyph != nil {
                            let fallback = ImageRenderer(content: Badge(symbol: "desktopcomputer")
                                .environment(\.dynamicTypeSize, size))
                            fallback.proposedSize = ProposedViewSize(width: width, height: nil)
                            fallback.scale = 1
                            let symbol = try #require(fallback.uiImage)
                            #expect(image.pngData() == symbol.pngData(),
                                    "A crowded badge must render its native symbol, rather than an empty fallback")
                        }
                        let lineHeight = UIFont.preferredFont(forTextStyle: .caption2,
                            compatibleWith: UITraitCollection(preferredContentSizeCategory: category)).lineHeight
                        // Production reserves two points around the line plus a
                        // five-point outer inset, on both vertical edges.
                        #expect(image.size.height <= ceil(lineHeight) + 2 * (2 + 5),
                                "The bounded badge must remain one rendered line at \(size)")
                    }
                }
            }
        }
    }
}
