import Foundation
import MoldClient

/// Something a machine could not do, in the terms the person used.
struct HostFailure: Identifiable, Equatable {
    let id = UUID()
    let host: MoldHost.ID
    /// Why it is keyed: a drain that retries four times must leave one line,
    /// not four.
    let verb: String
    let sentence: String
    let at: Date

    /// The verb every `.unreachable` failure collapses to. A machine that
    /// cannot be reached fails everything for the same one reason, so it
    /// gets the same one line -- and clears the same way, the moment
    /// `HostStore+Reachability` finds it answering again.
    static let reachVerb = "reach"
}
