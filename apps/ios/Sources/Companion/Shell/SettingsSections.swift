import MoldClient
import SwiftUI

/// Library, Notifications and Live Activities (DESIGN.md §5.6), between
/// Machines and About.
struct SettingsSections: View {
    @AppStorage(Preference.autoSaveToPhotos) private var autoSave = false
    @AppStorage(Preference.notifyFinished) private var finished = true
    @AppStorage(Preference.notifyFailed) private var failed = true
    @AppStorage(Preference.notifyHeld) private var held = true
    @AppStorage(Preference.liveActivities) private var live = true
    @Environment(Notifier.self) private var notifier
    @Bindable var access: SettingsAccess

    var body: some View {
        OfflineLibrarySection(autoSave: Binding(get: { autoSave }, set: { enabled in
            if !enabled { autoSave = false; return }
            guard !access.requestingPhotos else { return }
            access.requestingPhotos = true
            Task {
                defer { access.requestingPhotos = false }
                access.photosStatus = await PhotosAccess.request()
                autoSave = PhotosAccess.canSave(access.photosStatus)
                access.recovery = PermissionRecovery.photos(access.photosStatus)
            }
        }), photosRecovery: PermissionRecovery.photos(access.photosStatus))
        Section {
            Toggle("Finished", isOn: $finished)
            Toggle("Didn't Finish", isOn: $failed)
            Toggle("Waiting on a Machine", isOn: $held)
            if !access.systemAllowsNotifications { PermissionSettingsButton(recovery: .notifications) }
            if access.notificationsNotRequested {
                Button("Enable Notifications") {
                    Task {
                        await notifier.requestAuthorization()
                        await access.refresh()
                        if !access.systemAllowsNotifications { access.recovery = .notifications }
                    }
                }
            }
        } header: {
            SectionHeader(String(localized: "Notifications"))
        } footer: {
            Text(access.systemAllowsNotifications
                 ? String(localized: "Sent when a render settles while Mold Studio is in the background.")
                 : String(localized: "Notifications are off for Mold Studio in the Settings app."))
                .foregroundStyle(.secondaryText)
        }
        if UIDevice.current.userInterfaceIdiom == .phone {
            Section {
                Toggle("Show Renders on the Lock Screen", isOn: $live)
                if !access.systemAllowsLiveActivities { PermissionSettingsButton(recovery: .liveActivities) }
            } header: {
                SectionHeader(String(localized: "Live Activities"))
            } footer: {
                Text("The server can't reach this iPhone directly, so a render followed from the Lock Screen may show as out of date until you open Mold Studio.")
                    .foregroundStyle(.secondaryText)
            }
        }
    }

}
