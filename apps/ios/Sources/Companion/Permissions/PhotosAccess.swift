import Photos

enum PhotosAccess {
    static func request() async -> PHAuthorizationStatus {
        let status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        return status == .notDetermined ? await PHPhotoLibrary.requestAuthorization(for: .addOnly) : status
    }
    static func canSave(_ status: PHAuthorizationStatus) -> Bool {
        status == .authorized || status == .limited
    }
}
