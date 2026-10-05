import ActivityKit
import Observation
import Photos
import UserNotifications

/// Owned by the persistent Settings screen, independently of lazy Form rows.
@Observable final class SettingsAccess {
    var systemAllowsNotifications = true
    var notificationsNotRequested = false
    var systemAllowsLiveActivities = true
    var photosStatus = PHPhotoLibrary.authorizationStatus(for: .addOnly)
    var recovery: PermissionRecovery?
    var requestingPhotos = false

    func refresh() async {
        photosStatus = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        systemAllowsNotifications = settings.authorizationStatus != .denied
        notificationsNotRequested = settings.authorizationStatus == .notDetermined
        systemAllowsLiveActivities = ActivityAuthorizationInfo().areActivitiesEnabled
    }
}
