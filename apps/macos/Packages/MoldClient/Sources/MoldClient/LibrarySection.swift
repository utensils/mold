import Foundation

/// A day's worth of prints.
public struct LibrarySection: Identifiable, Hashable, Sendable {
    public let id: String
    public let day: Date
    public let items: [LibraryEntry]
}

public enum LibraryGrouping {
    /// Groups prints into day sections, newest day first and newest print
    /// first within a day.
    ///
    /// Grouping happens in the caller's calendar and time zone, so "Today"
    /// means the viewer's today -- a print made at 23:50 in another zone still
    /// files under the day the person who is looking at it would call it.
    public static func byDay(
        _ items: [LibraryEntry],
        calendar: Calendar = .current
    ) -> [LibrarySection] {
        let buckets = Dictionary(grouping: items) { item in
            calendar.startOfDay(for: item.createdAt)
        }
        return buckets.keys.sorted(by: >).map { day in
            LibrarySection(
                id: ISO8601DateFormatter().string(from: day),
                day: day,
                items: buckets[day]!.sorted { $0.print.timestamp > $1.print.timestamp }
            )
        }
    }

    /// The heading a day gets. Today and Yesterday are named; older days get a
    /// date, and a day in a previous year says which year.
    public static func title(
        for day: Date,
        now: Date = .now,
        calendar: Calendar = .current
    ) -> String {
        if calendar.isDateInToday(day) { return "Today" }
        if calendar.isDateInYesterday(day) { return "Yesterday" }

        let formatter = DateFormatter()
        formatter.calendar = calendar
        let sameYear = calendar.component(.year, from: day) == calendar.component(.year, from: now)
        formatter.setLocalizedDateFormatFromTemplate(sameYear ? "EEEEdMMMM" : "dMMMMyyyy")
        return formatter.string(from: day)
    }
}
