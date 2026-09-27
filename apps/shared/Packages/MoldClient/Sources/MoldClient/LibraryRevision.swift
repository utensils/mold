import Foundation
import Observation

/// How many times the rows have changed.
///
/// A counter rather than a comparison, because two libraries of the same size
/// differ by one print's favourite star and comparing them costs exactly what
/// the index it feeds exists to avoid. Its own type so the store holds one
/// reference instead of a stored counter, a bump and the paragraph explaining
/// them -- and so anything else deriving from the rows can be handed the same
/// signal without reaching into the store.
@MainActor
@Observable
public final class LibraryRevision {
    public private(set) var value = 0

    public init() {}

    public func bump() { value &+= 1 }
}
