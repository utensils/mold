import Foundation
import Testing
import SwiftUI

@testable import MoldStyle

@Test func theBoardAlternatesAndStartsLight() {
    let squares = AlphaBed.squares(in: CGRect(x: 0, y: 0, width: 16, height: 16))
    #expect(squares.count == 4)
    #expect(squares.map(\.isLight) == [true, false, false, true])
    #expect(squares.allSatisfy { $0.rect.width == AlphaBed.cell })
}

@Test func theBoardIsClippedToThePicture() {
    let rect = CGRect(x: 10, y: 20, width: 20, height: 9)
    let squares = AlphaBed.squares(in: rect)
    // 3 columns x 2 rows, the last of each clipped rather than overhanging.
    #expect(squares.count == 6)
    #expect(squares.allSatisfy { rect.contains($0.rect) })
    let area = squares.reduce(CGFloat(0)) { $0 + $1.rect.width * $1.rect.height }
    #expect(abs(area - 20 * 9) < 0.001)
}

@Test func anEmptyRectDrawsNothing() {
    #expect(AlphaBed.squares(in: .zero).isEmpty)
}

@Test func theBoardSitsExactlyUnderAFittedPicture() {
    // A 2:1 picture in a square view letterboxes top and bottom.
    let fitted = AlphaBed.fittedRect(
        content: CGSize(width: 200, height: 100), in: CGRect(x: 0, y: 0, width: 400, height: 400))
    #expect(fitted == CGRect(x: 0, y: 100, width: 400, height: 200))
    #expect(AlphaBed.fittedRect(content: .zero, in: CGRect(x: 0, y: 0, width: 10, height: 10)) == .zero)
}
