import Foundation
import Testing

@testable import MoldClient

/// Midday on a fixed date.
///
/// Deliberately NOT relative to `now`: a test that builds "30 hours ago" from
/// the wall clock groups into two days or three depending on the time of day it
/// runs, and duly failed the first time it ran after midnight.
private let noon = UInt64(1_789_560_000)

private func item(_ secondsAgo: Int, host: UUID = UUID()) -> LibraryEntry {
    let stamp = noon - UInt64(secondsAgo)
    let meta = try! MoldJSON.decoder.decode(OutputMetadata.self, from: Data("{}".utf8))
    return LibraryEntry(
        host: MoldHost(id: host, name: "h", baseURL: URL(string: "http://h")!),
        print: GalleryPrint(
            filename: "f-\(secondsAgo)-\(host).png", metadata: meta, timestamp: stamp,
            format: "png", sizeBytes: 1, mediaVersion: "v", title: nil, tags: nil,
            favorite: nil, collections: nil, trashedAt: nil, purgeAt: nil
        )
    )
}

@Test func groupsPrintsIntoDaysNewestFirst() {
    // From midday, 30 hours back is the previous day whatever the clock says.
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
    // Anchored to the start of today so the assertion cannot straddle
    // midnight the way a bare `.now` minus 24h can.
    let today = Calendar.current.startOfDay(for: .now)
    #expect(LibraryGrouping.title(for: today) == "Today")
    #expect(LibraryGrouping.title(for: today.addingTimeInterval(-86_400)) == "Yesterday")
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
