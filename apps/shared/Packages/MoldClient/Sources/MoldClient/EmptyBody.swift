import Foundation

/// `{}`.
///
/// Its own file rather than one route group's: the chain's cancel and
/// resume, a config delete and a pairing session all post it, and a shared
/// helper living inside whichever group happened to need it first is a
/// helper nobody else can find.
struct EmptyBody: Encodable {}
