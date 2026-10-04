import Foundation
import MoldClient
import Testing

@testable import MoldCompanion

@MainActor
struct OfflineLibraryNoticeTests {
    @Test func noOfflineHostsProduceNoPinnedNotice() {
        #expect(OfflineLibraryNotice.summary(hosts: []) == nil)
    }

    @Test func summaryCountsDuplicateNamesAndDoesNotGrowWithNames() throws {
        func hosts(_ name: String) -> [MoldHost] {
            (1...4).map { MoldHost(name: name, baseURL: URL(string: "http://fixture-\($0):7680")!) }
        }
        let short = try #require(OfflineLibraryNotice.summary(hosts: hosts("Fixture")))
        let long = try #require(OfflineLibraryNotice.summary(hosts: hosts(String(repeating: "Long machine name ", count: 100))))
        #expect(short == long)
        #expect(short.contains("4"))
        #expect(short.localizedCaseInsensitiveContains("saved prints"))
        #expect(short.count < 60)
        let one = try #require(OfflineLibraryNotice.summary(hosts: [hosts("Fixture")[0]]))
        #expect(one.contains("1"))
        #expect(one != short)
    }
}
