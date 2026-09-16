import Foundation
import Testing

@testable import MoldClient

private func item(_ secondsAgo: Int, host: UUID = UUID()) -> LibraryEntry {
    let stamp = UInt64(Date.now.timeIntervalSince1970) - UInt64(secondsAgo)
    let meta = try! MoldJSON.decoder.decode(OutputMetadata.self, from: Data("{}".utf8))
    return LibraryEntry(
        hostID: host, hostName: "h",
        print: GalleryPrint(
            filename: "f-\(secondsAgo)-\(host).png", metadata: meta, timestamp: stamp,
            format: "png", sizeBytes: 1, mediaVersion: "v", title: nil, tags: nil,
            favorite: nil, collections: nil, trashedAt: nil, purgeAt: nil
        )
    )
}

@Test func groupsPrintsIntoDaysNewestFirst() {
    let sections = LibraryGrouping.byDay([item(0), item(60 * 60 * 30), item(120)])

    #expect(sections.count == 2)
    #expect(sections[0].items.count == 2)      // today
    #expect(sections[1].items.count == 1)      // yesterday-ish
    #expect(sections[0].day > sections[1].day)
}

@Test func newestPrintLeadsItsDay() {
    let sections = LibraryGrouping.byDay([item(600), item(60)])
    let day = try! #require(sections.first)
    #expect(day.items[0].print.timestamp > day.items[1].print.timestamp)
}

@Test func todayAndYesterdayAreNamedRatherThanDated() {
    #expect(LibraryGrouping.title(for: .now) == "Today")
    #expect(LibraryGrouping.title(for: .now.addingTimeInterval(-86_400)) == "Yesterday")
}

@Test func anOlderDayGetsADateAndAPreviousYearSaysWhichYear() {
    let now = Date(timeIntervalSince1970: 1_789_500_000)
    let lastYear = now.addingTimeInterval(-400 * 86_400)
    let title = LibraryGrouping.title(for: lastYear, now: now)
    #expect(title != "Today" && title != "Yesterday")
    // A print from a previous year is ambiguous without the year on it.
    let year = Calendar.current.component(.year, from: lastYear)
    #expect(title.contains(String(year)))
}

@Test func printsFromDifferentHostsShareADaySection() {
    let a = UUID(), b = UUID()
    let sections = LibraryGrouping.byDay([item(30, host: a), item(60, host: b)])
    // One merged timeline is the point: a day holds whatever was made that
    // day, on whichever machine.
    #expect(sections.count == 1)
    #expect(Set(sections[0].items.map(\.hostID)) == [a, b])
}
