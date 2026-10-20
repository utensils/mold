import SwiftUI

/// No stores, credentials, updater or network clients exist in this view.
struct StartupView: View {
    let failure: String?
    var homeAccess: StartupHomeAccess?
    var resolvedHome: URL?
    var body: some View {
        VStack(spacing: 16) {
            if let failure {
                Text("Mold couldn't start").font(.title2)
                Text(failure).multilineTextAlignment(.center).textSelection(.enabled)
            } else {
                ProgressView()
                Text("Starting Mold…").font(.title2)
                Text("Preparing this Mac's library and engine.").foregroundStyle(.secondary)
            }
            if let homeAccess, let resolvedHome {
                Button("Allow access to Mold folder…") { homeAccess.choose(existingHome: resolvedHome) }
                    .disabled(homeAccess.isChoosing)
                if let notice = homeAccess.notice {
                    Text(notice).font(.callout).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center).textSelection(.enabled)
                }
            }
            Button("Quit Mold", action: { NSApp.terminate(nil) })
        }
        .padding(32)
        .frame(maxWidth: 540)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
