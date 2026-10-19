import Foundation

/// One sync record as hide-only privacy evidence: an origin print and the copy
/// a destination holds of it, with the routes and installation it was proven
/// through. Equality is the identity clock's whole fingerprint -- a copy
/// recorded without a destination host IS on the local host, so stamping that
/// host later is not a change, and per-record bookkeeping (organization,
/// pending flags, checkpoint tokens) never enters it.
public struct SyncPrivacyEdge: Hashable, Sendable {
    public let origin: PrintID
    public let copy: PrintID
    public let sourceRoute: String?
    public let destinationRoute: String?
    public let destinationInstance: String?

    public init(origin: PrintID, destinationHost: MoldHost.ID?, destinationFilename: String,
                localHost: MoldHost.ID, sourceRoute: String?, destinationRoute: String?,
                destinationInstance: String?) {
        self.origin = origin
        self.copy = PrintID(host: destinationHost ?? localHost, filename: destinationFilename)
        self.sourceRoute = sourceRoute
        self.destinationRoute = destinationRoute
        self.destinationInstance = destinationInstance
    }

    /// Whether both ends were proven through the routes and installation the
    /// machines answer on now. No authority admits everything.
    public func admitted(by authority: [MoldHost.ID: String]?) -> Bool {
        guard let authority else { return true }
        guard let current = authority[origin.host], sourceRoute == current,
              let destinationRoute else { return false }
        return authority[copy.host]
            == [copy.host.uuidString, destinationRoute, destinationInstance ?? ""].joined(separator: "\n")
    }

    /// Admitted edges compressed to one stable member per copy component.
    public static func components(_ edges: some Sequence<SyncPrivacyEdge>,
                                  authority: [MoldHost.ID: String]?) -> [PrintID: PrintID] {
        LibraryMerge.copyComponents(edges: edges.filter { $0.admitted(by: authority) }.map { ($0.origin, $0.copy) })
    }
}
