import Foundation

/// What to tell a person about a failure. Deliberately short -- it is read at
/// every site `HostStore.report` is called from, not written out as a stack
/// trace.
///
/// Internal, and app-side: MoldClient never needs to turn an error into
/// prose for a person, and a public extension on `Error` in a shared package
/// is a global nobody asked for.
extension Error {
    var sentence: String {
        (self as? LocalizedError)?.errorDescription ?? localizedDescription
    }
}
