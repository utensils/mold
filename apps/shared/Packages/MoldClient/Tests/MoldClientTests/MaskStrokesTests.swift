import Foundation
import Testing

@testable import MoldClient

private func stroke(_ x: Double = 0, erases: Bool = false) -> MaskStroke {
    MaskStroke(points: [CGPoint(x: x, y: 0)], radius: MaskBrush.defaultSize, erases: erases)
}

@Test func aStrokePastTheLimitDropsTheOldestAndKeepsTheNewest() {
    var strokes = MaskStrokes()
    for index in 0 ..< (MaskStrokes.limit + 5) {
        strokes.add(stroke(Double(index)))
    }
    #expect(strokes.strokes.count == MaskStrokes.limit)
    // The oldest five (0...4) are gone; the newest is still there.
    #expect(strokes.strokes.first?.points.first?.x == CGFloat(5))
    #expect(strokes.strokes.last?.points.first?.x == CGFloat(MaskStrokes.limit + 4))
}

@Test func undoOnAnEmptyStackAnswersFalseAndChangesNothing() {
    var strokes = MaskStrokes()
    #expect(strokes.undo() == false)
    #expect(strokes.isEmpty)

    strokes.add(stroke())
    #expect(strokes.undo() == true)
    #expect(strokes.strokes.isEmpty)
    #expect(strokes.undo() == false)
}

@Test func invertIsItsOwnInverse() {
    var strokes = MaskStrokes()
    #expect(strokes.isInverted == false)
    strokes.invert()
    #expect(strokes.isInverted == true)
    strokes.invert()
    #expect(strokes.isInverted == false)
}

@Test func clearLeavesAnEmptyStackThatStillKnowsItIsNotInverted() {
    var strokes = MaskStrokes()
    strokes.add(stroke())
    strokes.add(stroke(1))
    strokes.clear()
    #expect(strokes.strokes.isEmpty)
    #expect(strokes.isInverted == false)
    #expect(strokes.isEmpty)
}

@Test func theBrushStepsThroughTheListAndStopsAtEachEnd() {
    #expect(MaskBrush.stepped(MaskBrush.defaultSize, by: 1) == 64)
    #expect(MaskBrush.stepped(MaskBrush.defaultSize, by: -1) == 16)
    #expect(MaskBrush.stepped(MaskBrush.sizes.first!, by: -1) == MaskBrush.sizes.first!)
    #expect(MaskBrush.stepped(MaskBrush.sizes.last!, by: 1) == MaskBrush.sizes.last!)
}
