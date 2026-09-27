import Foundation
import Testing

@testable import MoldClient

/// What to call each file when several prints land in one folder.
///
/// **Fails today**: the multi-print save `removeItem`s at the destination and
/// copies over it (`LibraryActions.swift:100-101`), so a file the person
/// already had is destroyed with no prompt, and two prints whose names differ
/// only by case collapse onto one file on the case-insensitive volume APFS is
/// by default. There is nothing that decides a name.
@Suite struct SaveNamesSuite {

    @Test func aFreeNameIsKeptAsItIs() {
        var names = SaveNames()
        #expect(names.claim("robot.png") == "robot.png")
    }

    @Test func anameAlreadyInTheFolderIsNotWrittenOver() {
        var names = SaveNames(existing: ["robot.png"])
        #expect(names.claim("robot.png") == "robot 2.png")
    }

    @Test func aRunOfCollisionsCountsUp() {
        var names = SaveNames(existing: ["robot.png", "robot 2.png"])
        #expect(names.claim("robot.png") == "robot 3.png")
        #expect(names.claim("robot.png") == "robot 4.png")
    }

    /// Two prints in ONE selection: `Robot.png` and `robot.png` are one file
    /// on a case-insensitive volume, so the second must not silently replace
    /// the first.
    @Test func twoPrintsThatDifferOnlyByCaseBothLand() {
        var names = SaveNames()
        #expect(names.claim("Robot.png") == "Robot.png")
        #expect(names.claim("robot.png") == "robot 2.png")
    }

    /// APFS is normalization-insensitive too: NFC and NFD café are one file.
    @Test func twoPrintsThatDifferOnlyByNormalizationBothLand() {
        var names = SaveNames()
        let composed = "caf\u{00E9}.png"
        let decomposed = "cafe\u{0301}.png"
        #expect(names.claim(composed) == composed)
        #expect(names.claim(decomposed) != decomposed)
    }

    @Test func anExtensionlessNameStillCountsUp() {
        var names = SaveNames(existing: ["render"])
        #expect(names.claim("render") == "render 2")
    }

    @Test func aNameWithSeveralDotsKeepsItsLastExtension() {
        var names = SaveNames(existing: ["a.b.png"])
        #expect(names.claim("a.b.png") == "a.b 2.png")
    }
}
