import Foundation

/// How long a trashed print has left, as a NUMBER.
///
/// Three surfaces say it in three different ways -- the sentence under
/// Recently Deleted, the badge on a tile, and what VoiceOver reads -- and
/// each had worked the days out for itself, two of them by dividing by
/// 86,400 and one by asking the calendar. Different wording per surface is
/// the point; a second answer to "how many days" is not.
///
/// `purge_at` is derived by the host from the retention in force RIGHT NOW,
/// never stored -- so it moves when somebody changes the setting, and it is
/// the only number worth showing per print.
public enum TrashCountdown {
    /// Whole days from `now` until the host purges this print, NEGATIVE once
    /// it is due, and `nil` where the machine gave no purge date at all.
    ///
    /// CALENDAR days -- the count of midnights between here and there, in the
    /// viewer's own calendar -- not elapsed 86,400-second chunks and not
    /// `dateComponents` between two instants, which is the same elapsed count
    /// with daylight saving folded in. "Deleting in 2 days" is a promise
    /// about a DATE: at 23:00 on Monday, a purge at 01:00 on Tuesday is
    /// tomorrow, and both of the other two readings call it today.
    public static func days(until purgeAt: UInt64?, now: Date = .now,
                            calendar: Calendar = .current) -> Int? {
        guard let purgeAt else { return nil }
        let due = Date(timeIntervalSince1970: TimeInterval(purgeAt))
        // Already due is its own answer, before any rounding to a date.
        guard due > now else { return -1 }
        return calendar.dateComponents([.day],
                                       from: calendar.startOfDay(for: now),
                                       to: calendar.startOfDay(for: due)).day ?? 0
    }
}
