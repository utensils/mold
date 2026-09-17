import Foundation
import Testing

@testable import Mold

@MainActor
struct MaskUndoTests {
    @Test func theEnvironmentManagerWinsWhenThereIsOne() {
        let environment = UndoManager()
        let own = UndoManager()
        #expect(MaskUndo.resolve(environment: environment, own: own) === environment)
    }

    @Test func theSheetFallsBackToItsOwn() {
        let own = UndoManager()
        #expect(MaskUndo.resolve(environment: nil, own: own) === own)
    }
}
