import Testing

@testable import Mold

struct SyncProgressCopyTests {
    @Test func bannerNamesTheDestinationInsteadOfHere() {
        let line = SyncProgressCopy.checked(322, of: 585, copied: 23, present: 3_713, on: "hal9000", issues: 2)
        #expect(line == "Checked 322 of 585 · Copied 23 · 3,713 already on hal9000 · Issues 2")
        #expect(!line.contains("here"))
        #expect(SyncProgressCopy.checking(585, present: 3_713, on: "This Mac")
            == "Checking 585 changed prints · 3,713 already on This Mac")
    }
}
