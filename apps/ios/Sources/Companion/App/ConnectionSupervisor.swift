import SwiftUI

/// Follows the scene's phase: live streams in the foreground, none in the
/// background, and a full reconcile on the way back (DESIGN.md §A).
///
/// iOS suspends a backgrounded app's sockets without closing them, so a
/// stream left "open" would look alive while delivering nothing. Stopping on
/// the way out and re-asking on the way in is the only honest model.
struct ConnectionSupervisor: ViewModifier {
    @Environment(\.scenePhase) private var phase
    let stores: CompanionStores

    func body(content: Content) -> some View {
        content
            .task { await stores.becameActive() }
            .task(id: phase) {
                guard phase == .active else { return }
                for await _ in NetworkRouteChanges.stream() {
                    do { try await Task.sleep(for: .milliseconds(500)) } catch { return }
                    guard !Task.isCancelled else { return }
                    await stores.hosts.refreshPairedRoutes()
                }
            }
            .onChange(of: phase) { _, new in
                switch new {
                case .active: Task { await stores.becameActive() }
                case .background: stores.enteredBackground()
                default: break
                }
            }
    }
}

extension View {
    func supervisesConnections(_ stores: CompanionStores) -> some View {
        modifier(ConnectionSupervisor(stores: stores))
    }
}
