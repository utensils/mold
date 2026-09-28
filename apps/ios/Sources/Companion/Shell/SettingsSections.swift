import ActivityKit
import MoldClient
import SwiftUI
import UserNotifications

/// Library, Notifications and Live Activities (DESIGN.md §5.6), between
/// Machines and About.
struct SettingsSections: View {
    @Environment(ThumbnailLoader.self) private var thumbnails
    @AppStorage(Preference.autoSaveToPhotos) private var autoSave = false
    @AppStorage(Preference.notifyFinished) private var finished = true
    @AppStorage(Preference.notifyFailed) private var failed = true
    @AppStorage(Preference.notifyHeld) private var held = true
    @AppStorage(Preference.liveActivities) private var live = true
    @State private var systemAllowsNotifications = true

    var body: some View {
        OfflineLibrarySection(autoSave: $autoSave)
            // On a real row's section: a `.task` on EmptyView never runs.
            .task {
                let settings = await UNUserNotificationCenter.current().notificationSettings()
                systemAllowsNotifications = settings.authorizationStatus != .denied
            }
        Section {
            Toggle("Finished", isOn: $finished)
            Toggle("Didn't Finish", isOn: $failed)
            Toggle("Waiting on a Machine", isOn: $held)
        } header: {
            SectionHeader(String(localized: "Notifications"))
        } footer: {
            Text(systemAllowsNotifications
                 ? String(localized: "Sent when a render settles while Mold Studio is in the background.")
                 : String(localized: "Notifications are off for Mold Studio in the Settings app."))
                .foregroundStyle(.secondaryText)
        }
        if UIDevice.current.userInterfaceIdiom == .phone {
            Section {
                Toggle("Show Renders on the Lock Screen", isOn: $live)
            } header: {
                SectionHeader(String(localized: "Live Activities"))
            } footer: {
                Text("The server can't reach this iPhone directly, so a render followed from the Lock Screen may show as out of date until you open Mold Studio.")
                    .foregroundStyle(.secondaryText)
            }
        }
    }
}
