import SwiftUI

/// Settings ▸ General -- what to do while Mold is in the background.
///
/// Two toggles, minimal on purpose: M7 curates this tab. Both persist through
/// `AppStorageSuite` so `MOLD_NATIVE_FRESH` swaps them the way it swaps every
/// other preference.
struct GeneralSettings: View {
    @AppStorage("badgeLandedPrints", store: AppStorageSuite.defaults)
    private var badgeLandedPrints = true
    @AppStorage("notifyRenders", store: AppStorageSuite.defaults)
    private var notifyRenders = true

    var body: some View {
        Form {
            Section {
                Toggle("Badge the Dock icon with prints that arrive while Mold is in the background",
                       isOn: $badgeLandedPrints)
                Toggle("Notify when a render finishes or a job fails", isOn: $notifyRenders)
            } footer: {
                Text("Only while Mold is in the background. Coming back to the app clears the badge.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}
