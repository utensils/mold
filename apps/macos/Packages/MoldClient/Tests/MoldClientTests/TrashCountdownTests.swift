import Foundation
import Testing

@testable import MoldClient

/// How long a trashed print has left, as a number three surfaces word three
/// different ways.
///
/// **Fails today**: the answer was `dateComponents([.day], from: now, to: due)`
/// between two INSTANTS, which is whole elapsed days -- so a purge at 01:00 on
/// Tuesday, read at 23:00 on Monday, answered 0 and the tile said "today"
/// about tomorrow. The doc comment, the commit message and the ledger all
/// claimed calendar days.
@Suite struct TrashCountdownSuite {
    /// A fixed zone, so the boundaries below are the same boundaries
    /// everywhere this runs. London, because it has a DST transition the test
    /// below stands on.
    private var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "Europe/London")!
        return calendar
    }

    private func moment(_ text: String) -> Date {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return formatter.date(from: text)!
    }

    private func seconds(_ text: String) -> UInt64 {
        UInt64(moment(text).timeIntervalSince1970)
    }

    /// Two hours apart, across midnight: one sleep away, so one day.
    @Test func twoHoursAcrossMidnightIsTomorrowNotToday() {
        #expect(TrashCountdown.days(until: seconds("2026-03-03 01:00"),
                                    now: moment("2026-03-02 23:00"),
                                    calendar: calendar) == 1)
    }

    /// Twenty-three hours apart, same day: still today.
    @Test func almostAWholeDayInsideOneDayIsStillToday() {
        #expect(TrashCountdown.days(until: seconds("2026-03-02 23:59"),
                                    now: moment("2026-03-02 00:01"),
                                    calendar: calendar) == 0)
    }

    /// The clocks go forward at 01:00 on 29 March 2026, so that day is 23
    /// hours long. Two midnights away is two days, whatever the seconds say --
    /// an elapsed-seconds count answers 1 here.
    @Test func aShortDayIsStillOneDay() {
        #expect(TrashCountdown.days(until: seconds("2026-03-30 00:30"),
                                    now: moment("2026-03-28 01:00"),
                                    calendar: calendar) == 2)
    }

    /// Already due is its own answer, before any rounding to a date -- it is
    /// what `TrashRetention` says "Deleting soon" about.
    @Test func alreadyDueIsNegative() {
        #expect(TrashCountdown.days(until: seconds("2026-03-02 09:00"),
                                    now: moment("2026-03-02 10:00"),
                                    calendar: calendar) == -1)
    }

    @Test func noPurgeDateIsNoAnswer() {
        #expect(TrashCountdown.days(until: nil) == nil)
    }
}
