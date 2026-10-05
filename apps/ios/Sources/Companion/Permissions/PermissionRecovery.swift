import AVFoundation
import Photos
import SwiftUI
import UIKit

/// Recovery after a contextual system request. Restricted access cannot be
/// repaired by promising a toggle: Screen Time or device management may own it.
struct PermissionRecovery: Equatable {
    let title: String
    let message: String
    let settingsURL: URL?

    static func photos(_ status: PHAuthorizationStatus) -> Self? {
        switch status {
        case .denied:
            Self(title: String(localized: "Allow Saving to Photos"),
                 message: String(localized: "To save pictures and videos, allow Mold Studio to add photos in Settings. Then return and tap Save to Photos again."),
                 settingsURL: appSettings)
        case .restricted:
            restricted(String(localized: "Photos"))
        default: nil
        }
    }

    static func camera(_ status: AVAuthorizationStatus) -> Self? {
        switch status {
        case .denied:
            Self(title: String(localized: "Allow Camera Access"),
                 message: String(localized: "To take a photo or scan a pairing code, enable Camera for Mold Studio in Settings."),
                 settingsURL: appSettings)
        case .restricted: restricted(String(localized: "Camera"))
        default: nil
        }
    }

    static var localNetwork: Self {
        Self(title: String(localized: "Allow Local Network Access"),
             message: String(localized: "To find and connect to nearby machines, enable Local Network for Mold Studio in Settings."),
             settingsURL: appSettings)
    }
    static var notifications: Self {
        Self(title: String(localized: "Allow Notifications"),
             message: String(localized: "To receive render updates, turn on Allow Notifications for Mold Studio in Settings."),
             settingsURL: URL(string: UIApplication.openNotificationSettingsURLString))
    }
    static var liveActivities: Self {
        Self(title: String(localized: "Allow Live Activities"),
             message: String(localized: "To follow renders on the Lock Screen, enable Live Activities for Mold Studio in Settings."),
             settingsURL: appSettings)
    }
    private static var appSettings: URL? { URL(string: UIApplication.openSettingsURLString) }
    private static func restricted(_ resource: String) -> Self {
        Self(title: String(localized: "\(resource) Access Is Restricted"),
             message: String(localized: "Screen Time or device management may restrict \(resource) access. Contact the person who manages this device to allow access."),
             settingsURL: nil)
    }
}

private struct PermissionAlert: ViewModifier {
    @Binding var recovery: PermissionRecovery?
    @Environment(\.openURL) private var openURL
    func body(content: Content) -> some View {
        content.alert(recovery?.title ?? "", isPresented: Binding(
            get: { recovery != nil }, set: { if !$0 { recovery = nil } }), presenting: recovery) { item in
                if let url = item.settingsURL {
                    Button("Open Settings") { openURL(url) }
                    Button("Not Now", role: .cancel) {}
                } else { Button("OK", role: .cancel) {} }
            } message: { item in Text(item.message) }
    }
}

extension View {
    func permissionAlert(_ recovery: Binding<PermissionRecovery?>) -> some View {
        modifier(PermissionAlert(recovery: recovery))
    }
}

struct PermissionSettingsButton: View {
    let recovery: PermissionRecovery
    @State private var presented: PermissionRecovery?
    var body: some View {
        Button(recovery.title) { presented = recovery }
            .permissionAlert($presented)
    }
}
