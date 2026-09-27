import Foundation
import Testing

@testable import MoldClient

/// What a print announces to VoiceOver.
///
/// The rule: a tile says everything it DRAWS. It draws a star, a play badge, a
/// machine name and a countdown, and a sighted person reads all four at a
/// glance -- so the label carries all four rather than the prompt alone.
@Suite struct AccessibilitySuite {
    let workstation = UUID()

    @Test func aPictureLeadsWithWhatItIsOf() {
        let entry = PrintFixtures.entry("a.png", host: workstation, prompt: "a brass diving helmet")
        #expect(entry.spokenDescription(showsHost: false) == "a brass diving helmet, picture")
    }

    /// A print with no prompt still has to be called something.
    @Test func aPrintWithNoPromptIsCalledByItsName() {
        let entry = PrintFixtures.entry("a.png", host: workstation)
        #expect(entry.spokenDescription(showsHost: false) == "a.png, picture")
    }

    @Test func aTitleWinsOverThePrompt() {
        let entry = PrintFixtures.entry("a.png", host: workstation, title: "Helmet",
                                        prompt: "a brass diving helmet")
        #expect(entry.spokenDescription(showsHost: false).hasPrefix("Helmet,"))
    }

    @Test func theKindIsSpokenNotInferredFromABadge() {
        let clip = PrintFixtures.entry("a.mp4", host: workstation, format: "mp4", prompt: "a hangar")
        let mesh = PrintFixtures.entry("a.glb", host: workstation, format: "glb", prompt: "a helmet")
        #expect(clip.spokenDescription(showsHost: false) == "a hangar, clip")
        #expect(mesh.spokenDescription(showsHost: false) == "a helmet, mesh")
    }

    @Test func aFavouriteSaysSo() {
        let entry = PrintFixtures.entry("a.png", host: workstation, favorite: true, prompt: "owls")
        #expect(entry.spokenDescription(showsHost: false) == "owls, picture, favourite")
    }

    /// Only when the grid is actually showing more than one machine -- the
    /// same rule the visible badge follows, because reading "on workstation" on
    /// every tile of a single-machine library is noise.
    @Test func theMachineIsNamedOnlyWhenTheBadgeWouldBe() {
        let entry = PrintFixtures.entry("a.png", host: workstation, hostName: "workstation", prompt: "owls")
        #expect(entry.spokenDescription(showsHost: true) == "owls, picture, on workstation")
        #expect(entry.spokenDescription(showsHost: false) == "owls, picture")
    }

    @Test func aTrashedPrintSaysHowLongItHas() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let print = PrintFixtures.print("a.png", prompt: "owls", trashedAt: 999_000,
                                        purgeAt: 1_000_000 + 86_400 * 3)
        let entry = LibraryEntry(host: MoldHost(id: workstation, name: "workstation", baseURL: URL(string: "http://h")!), print: print)
        #expect(entry.spokenDescription(showsHost: false, now: now)
            == "owls, picture, deleted, 3 days left")
    }

    @Test func aPrintPurgedTodaySaysToday() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let print = PrintFixtures.print("a.png", prompt: "owls", trashedAt: 999_000,
                                        purgeAt: 1_000_100)
        let entry = LibraryEntry(host: MoldHost(id: workstation, name: "workstation", baseURL: URL(string: "http://h")!), print: print)
        #expect(entry.spokenDescription(showsHost: false, now: now).hasSuffix("deleted, today"))
    }
}
