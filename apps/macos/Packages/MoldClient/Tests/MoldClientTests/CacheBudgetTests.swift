import Foundation
import Testing

@testable import MoldClient

/// Which cached files have to go to fit a budget.
@Suite struct CacheBudgetSuite {
    private func file(_ name: String, _ bytes: Int, _ used: TimeInterval)
        -> CacheBudget.File {
        CacheBudget.File(name: name, bytes: bytes, lastUsed: Date(timeIntervalSince1970: used))
    }

    @Test func nothingIsEvictedWhileItFits() {
        let files = [file("a", 100, 1), file("b", 100, 2)]
        #expect(CacheBudget.evictions(from: files, cap: 500).isEmpty)
    }

    @Test func theLeastRecentlyUsedGoesFirst() {
        let files = [file("new", 100, 30), file("old", 100, 10), file("mid", 100, 20)]
        #expect(CacheBudget.evictions(from: files, cap: 250) == ["old"])
    }

    @Test func itEvictsUntilItFitsAndNoFurther() {
        let files = [file("a", 100, 10), file("b", 100, 20), file("c", 100, 30)]
        #expect(CacheBudget.evictions(from: files, cap: 100) == ["a", "b"])
    }

    /// A cap of zero means "keep nothing", which is a real setting and not a
    /// disabled one -- someone with no disk to spare turns the cache off.
    @Test func aCapOfZeroEvictsEverything() {
        let files = [file("a", 100, 10), file("b", 100, 20)]
        #expect(Set(CacheBudget.evictions(from: files, cap: 0)) == ["a", "b"])
    }

    /// One file bigger than the whole budget cannot be kept, and saying so is
    /// better than evicting everything else and still not fitting.
    @Test func aFileLargerThanTheCapIsEvictedEvenIfItIsTheNewest() {
        let files = [file("small", 10, 10), file("huge", 1_000, 99)]
        #expect(CacheBudget.evictions(from: files, cap: 100) == ["huge"])
    }

    @Test func aFileExactlyAtTheCapIsKept() {
        #expect(CacheBudget.evictions(from: [file("a", 100, 1)], cap: 100).isEmpty)
    }
}
