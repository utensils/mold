import SwiftUI

/// A refused Library handoff leaves the existing draft intact until the user
/// explicitly chooses to keep working on it.
struct NativeVideoSourceFailure: View {
    let message: String
    let keepDraft: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(message).font(.callout).fixedSize(horizontal: false, vertical: true)
            Button("Keep current draft", action: keepDraft)
                .frame(minHeight: 44)
        }
    }
}
