import Testing
@testable import MoldClient

@Test func newMediaUsesPreviousVisitAndKeepsCurrentVisitStable() {
    var session = LibraryNewMedia()
    let first = session.beginVisit()
    #expect(!first.contains("old.png"))
    session.markSeen(["old.png", "old.png"])
    #expect(!first.contains("arrived.mp4"))
    let second = session.beginVisit()
    #expect(!second.contains("old.png"))
    #expect(second.contains("arrived.mp4"))
    session.markSeen(["old.png", "arrived.mp4"])
    #expect(second.contains("arrived.mp4"))
    let third = session.beginVisit()
    #expect(!third.contains("arrived.mp4"))
    #expect(third.contains("new.glb"))
}

@Test func firstVisitCanLoadItsBaselineLater() {
    var session = LibraryNewMedia()
    let first = session.beginVisit()
    session.markSeen(["loaded-later.png"])
    #expect(!first.contains("loaded-later.png"))
    #expect(!session.beginVisit().contains("loaded-later.png"))
}
