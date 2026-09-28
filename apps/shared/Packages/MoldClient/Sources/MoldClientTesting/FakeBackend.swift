import Foundation
import MoldClient
import Synchronization

/// A `MoldBackend` for store tests, built only on MoldClient's public API so
/// both native apps can use it (`MoldClientTesting`; test targets only).
///
/// Every route THROWS `FakeBackendError.unstubbed` until a test stubs it, so a
/// store that quietly calls a route its test never considered fails loudly
/// instead of receiving a made-up answer. Routes are named by their Swift
/// selector -- `"status()"`, `"batchStatus(id:)"`, `"export(_:format:)"` -- so
/// overloads stay distinct. Every call is recorded, in order, with its
/// arguments.
public final class FakeBackend: MoldBackend, Sendable {
    public let host: MoldHost

    public typealias Handler = @Sendable ([any Sendable]) async throws -> any Sendable
    public typealias StreamFactory = @Sendable ([any Sendable]) -> any Sendable

    struct State {
        var handlers: [String: Handler] = [:]
        var streams: [String: StreamFactory] = [:]
        var calls: [FakeCall] = []
    }

    let state = Mutex(State())

    public init(host: MoldHost = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation:7680")!)) {
        self.host = host
    }

    // MARK: Stubbing

    /// Answers `route` by running `handler` with the call's arguments.
    public func stub(_ route: String, _ handler: @escaping Handler) {
        state.withLock { $0.handlers[route] = handler }
    }

    /// Answers `route` with `value` every time.
    public func stub(_ route: String, returning value: some Sendable) {
        stub(route) { _ in value }
    }

    /// Makes `route` fail with `error` every time.
    public func stub(_ route: String, throwing error: some Error) {
        stub(route) { _ in throw error }
    }

    /// Answers a streaming route. Unstubbed streams finish at once, empty --
    /// a store must cope with a stream that ends, so that is the honest default.
    public func stubStream<Element: Sendable>(
        _ route: String, _ make: @escaping @Sendable ([any Sendable]) -> AsyncThrowingStream<Element, Error>
    ) {
        state.withLock { $0.streams[route] = { make($0) } }
    }

    // MARK: Inspecting

    public var calls: [FakeCall] { state.withLock { $0.calls } }

    /// How many times `route` was called.
    public func count(_ route: String) -> Int { calls.filter { $0.route == route }.count }

    // MARK: Plumbing the generated routes use

    func respond<T>(_ route: String, _ args: [any Sendable]) async throws -> T {
        let handler = state.withLock { state -> Handler? in
            state.calls.append(FakeCall(route: route, arguments: args))
            return state.handlers[route]
        }
        guard let handler else { throw FakeBackendError.unstubbed(route) }
        let value = try await handler(args)
        if T.self == Void.self { return () as! T }
        guard let typed = value as? T else {
            throw FakeBackendError.wrongType(route, expected: String(describing: T.self))
        }
        return typed
    }

    func stream<Element>(_ route: String, _ args: [any Sendable]) -> AsyncThrowingStream<Element, Error> {
        let factory = state.withLock { state -> StreamFactory? in
            state.calls.append(FakeCall(route: route, arguments: args))
            return state.streams[route]
        }
        if let made = factory?(args) as? AsyncThrowingStream<Element, Error> { return made }
        return AsyncThrowingStream { $0.finish() }
    }
}

/// One recorded call: the route's selector and what it was called with.
public struct FakeCall: Sendable {
    public let route: String
    public let arguments: [any Sendable]
}

public enum FakeBackendError: Error, Equatable, CustomStringConvertible {
    /// A route the test never stubbed was called.
    case unstubbed(String)
    /// A stub answered with a value of the wrong type for the route.
    case wrongType(String, expected: String)

    public var description: String {
        switch self {
        case let .unstubbed(route): "FakeBackend: \(route) was called but never stubbed"
        case let .wrongType(route, expected): "FakeBackend: \(route) was stubbed with something other than \(expected)"
        }
    }
}
