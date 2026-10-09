import SwiftUI

/// Completion reports wait for a native context menu to close. Presenting a
/// sheet during menu tracking would dismiss the menu even with a stable owner.
private struct DeferredMenuSheet<Sheet: View>: ViewModifier {
    @Binding var requested: Bool
    let sheet: () -> Sheet
    private let tracking = ContextMenuTracking.shared
    @State private var completionRevision = 0

    func body(content: Content) -> some View {
        let _ = completionRevision
        // Observe the request while evaluating the presenter, rather than
        // relying on SwiftUI to observe a custom Binding's lazy getter.
        let presented = tracking.shouldPresentSheet(requested)
        return content
            .sheet(isPresented: Binding(
                get: { presented },
                set: { requested = $0 }
            ), content: sheet)
            .onReceive(tracking.didFinishTracking) { completionRevision = $0 }
    }
}

extension View {
    func deferredMenuSheet<Sheet: View>(
        isPresented: Binding<Bool>, @ViewBuilder content: @escaping () -> Sheet
    ) -> some View {
        modifier(DeferredMenuSheet(requested: isPresented, sheet: content))
    }
}
