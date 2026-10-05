import AVFoundation
import Observation

@Observable final class CameraAccess {
    var status = AVCaptureDevice.authorizationStatus(for: .video)
    var recovery: PermissionRecovery?

    /// Call only when the person chooses Take Photo or opens the scanner.
    func request() async -> Bool {
        if AVCaptureDevice.authorizationStatus(for: .video) == .notDetermined {
            _ = await AVCaptureDevice.requestAccess(for: .video)
        }
        refresh()
        recovery = PermissionRecovery.camera(status)
        return status == .authorized
    }
    func refresh() { status = AVCaptureDevice.authorizationStatus(for: .video) }
}
