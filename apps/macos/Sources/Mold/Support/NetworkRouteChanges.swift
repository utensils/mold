import Foundation
import Network

/// One bounded stream of path changes; disposal stops the system monitor.
nonisolated enum NetworkRouteChanges {
    static func stream() -> AsyncStream<Void> {
        AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            let monitor = NWPathMonitor()
            monitor.pathUpdateHandler = { _ in continuation.yield(()) }
            continuation.onTermination = { _ in monitor.cancel() }
            monitor.start(queue: DispatchQueue(label: "mold.connection-path"))
        }
    }
}
