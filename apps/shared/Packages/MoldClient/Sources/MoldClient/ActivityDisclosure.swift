import Foundation

/// Dismissing presentation never changes the operation or issue acknowledgment.
public struct ActivityDisclosure: Sendable {
    private var dismissed: Set<String> = []
    public init() {}
    public func shows(_ identity: String) -> Bool { !dismissed.contains(identity) }
    public var hasDismissed: Bool { !dismissed.isEmpty }
    public mutating func dismiss(_ identity: String) { dismissed.insert(identity) }
    public mutating func restore() { dismissed.removeAll() }
}
