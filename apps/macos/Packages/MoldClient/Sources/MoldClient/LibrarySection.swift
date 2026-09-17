import Foundation

/// A day's worth of prints, or -- when the order is not chronological -- the
/// whole list in one piece.
public struct LibrarySection: Identifiable, Hashable, Sendable {
    public let id: String
    /// The day this section heads, or `nil` when the list is not cut into
    /// days. "Today" over a Largest First run would be a lie about the row
    /// under it, and a day can only be one contiguous run when the order is
    /// chronological.
    public let day: Date?
    public let items: [LibraryEntry]
}

public extension LibrarySort {
    /// Whether a day heading describes the list this sort produces.
    var groupsByDay: Bool {
        switch self {
        case .newest, .oldest: true
        case .largest, .name: false
        }
    }
}

public enum LibraryGrouping {
    /// Cuts an ALREADY ORDERED list into day sections, keeping that order.
    ///
    /// It used to re-sort -- days descending, and newest-first inside each day
    /// -- which made the grid's order its own rather than the query's, so Sort
    /// By was a no-op on screen (`LibraryQuery.sorted` is correct and tested;
    /// it simply never reached the pixels). Worse, the viewer's ← → walk
    /// `visible` while the grid's cursor walks these sections, so under Oldest
    /// First the two moved in opposite directions over the same list. Sorting
    /// is `LibraryQuery`'s job and this is the cut, so `sections.flatMap(items)
    /// == visible` for every sort.
    ///
    /// Grouping happens in the caller's calendar and time zone, so "Today"
    /// means the viewer's today -- a print made at 23:50 in another zone still
    /// files under the day the person who is looking at it would call it.
    public static func byDay(
        _ items: [LibraryEntry],
        calendar: Calendar = .current
    ) -> [LibrarySection] {
        var days: [Date] = []
        var buckets: [Date: [LibraryEntry]] = [:]
        for item in items {
            let day = calendar.startOfDay(for: item.createdAt)
            if buckets[day] == nil { days.append(day) }
            buckets[day, default: []].append(item)
        }
        return days.map { day in
            // The day itself as the id: stable, unique per section, and free.
            // An `ISO8601DateFormatter()` here was one formatter ALLOCATED per
            // section, on every pass of the pane's body.
            LibrarySection(id: String(day.timeIntervalSince1970), day: day,
                           items: buckets[day] ?? [])
        }
    }

    /// The list in one piece, for an order days cannot describe.
    public static func ungrouped(_ items: [LibraryEntry]) -> [LibrarySection] {
        items.isEmpty ? [] : [LibrarySection(id: "all", day: nil, items: items)]
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
