import AVFoundation
import Network
import Photos
import Testing
import UIKit
@testable import MoldCompanion

@MainActor struct PermissionRecoveryTests {
    @Test func existingAutoSavePreferenceCanRequestMissingPhotosAccess() {
        #expect(PhotosAccess.needsRequest(autoSave: true, status: .notDetermined))
        #expect(!PhotosAccess.needsRequest(autoSave: false, status: .notDetermined))
        for status in [PHAuthorizationStatus.denied, .restricted, .authorized, .limited] {
            #expect(!PhotosAccess.needsRequest(autoSave: true, status: status))
        }
    }
    @Test func photosDeniedAndRestrictedHaveDifferentRecovery() {
        #expect(PermissionRecovery.photos(.denied)?.settingsURL == URL(string: UIApplication.openSettingsURLString))
        #expect(PermissionRecovery.photos(.restricted)?.settingsURL == nil)
        for status in [PHAuthorizationStatus.authorized, .limited, .notDetermined] {
            #expect(PermissionRecovery.photos(status) == nil)
        }
    }
    @Test func cameraRestrictedDoesNotPromiseASettingsToggle() {
        #expect(PermissionRecovery.camera(.denied)?.settingsURL != nil)
        #expect(PermissionRecovery.camera(.restricted)?.settingsURL == nil)
        #expect(PermissionRecovery.camera(.authorized) == nil)
        #expect(PermissionRecovery.camera(.notDetermined) == nil)
    }
    @Test func onlyPolicyDenialIsLocalNetworkPermissionFailure() {
        #expect(NearbyBrowser.isPermissionDenied(.dns(-65570)))
        #expect(!NearbyBrowser.isPermissionDenied(.dns(-65563)))
        #expect(!NearbyBrowser.isPermissionDenied(.posix(.ENETDOWN)))
    }
    @Test func notificationRecoveryUsesPublicNotificationSettingsURL() {
        #expect(PermissionRecovery.notifications.settingsURL == URL(string: UIApplication.openNotificationSettingsURLString))
    }
}
