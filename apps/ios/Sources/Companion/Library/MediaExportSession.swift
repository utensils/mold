import Foundation
import MoldClient
import Photos
import SwiftUI

@Observable
final class MediaExportSession: Identifiable {
    let id = UUID()
    let entry: LibraryEntry
    var options: ExportOptions?
    var format = "gif"
    var playback = GifPlayback.loop
    var repeatMode = GifRepeat.forever
    var pauseText = "0"
    var maxDimension = 720
    var fps = 12
    var frames = 36
    var transparent = UserDefaults.standard.bool(forKey: "turntable-transparent")
    var geometry: MeshExportGeometry?
    var destination = ExportDestination.share
    var error: String?
    var loading = true
    var converting = false
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private weak var actions: PrintActions?
    @ObservationIgnored private var cancelled = false
    @ObservationIgnored private let requestPhotosAccess: @MainActor () async -> PHAuthorizationStatus
    init(entry: LibraryEntry, actions: PrintActions,
         requestPhotosAccess: @escaping @MainActor () async -> PHAuthorizationStatus = { await PhotosAccess.request() }) {
        self.entry = entry; self.actions = actions; self.requestPhotosAccess = requestPhotosAccess
        if kind == .mesh { maxDimension = 512; fps = 10 }
    }
    var kind: MediaExportKind? { MediaExportKind(filename: entry.print.filename, trashed: entry.print.trashedAt != nil) }
    var meshCapabilities: MeshCapabilities? { actions?.hosts.capabilities[entry.hostID]?.mesh }
    var formats: [String] {
        kind == .mesh ? (meshCapabilities?.exportFormats ?? []).filter { $0 != "glb" && ["obj", "zip", "stl", "ply", "gif", "apng", "webp"].contains($0) } : options?.forVideo ?? []
    }
    var animation: Bool { MeshExport.isAnimated(format) }
    var isGif: Bool { format == "gif" }
    var pauseControl: GifPauseControl? { options?.gifPause.flatMap { $0.valid ? $0 : nil } }
    var takesPause: Bool { isGif && (playback == .bounce || repeatMode == .forever) && pauseControl != nil }
    var frameLimit: Int { MeshTurntableOptions.maximumFrames(atDimension: maxDimension, transparent: transparent) }
    var valid: Bool {
        guard formats.contains(format), !loading, !converting else { return false }
        if isGif {
            guard options?.playbackChoices.contains(playback) == true, options?.repeatChoices.contains(repeatMode) == true else { return false }
        }
        if kind == .mesh && animation { guard frames >= 8, frames <= frameLimit else { return false } }
        if takesPause { guard let pause = Int(pauseText), pauseControl?.accepts(pause) == true else { return false } }
        if let geometry, let limits = meshCapabilities?.exportGeometry?.sizeMm, let size = geometry.sizeMm {
            guard size.isFinite, size >= limits.min, size <= limits.max else { return false }
        }
        return true
    }
    var destinations: [ExportDestination] {
        // GIF and JPEG/PNG stills are PhotoKit resources; do not promise animated APNG/WebP preservation.
        format == "gif" ? [.share, .folder, .files, .photos] : [.share, .folder, .files]
    }
    func selectFormat() {
        geometry = actions?.hosts.capabilities[entry.hostID]?.meshGeometryDefaults(for: format)
        if !destinations.contains(destination) { destination = .share }
    }
    func load() {
        task?.cancel(); cancelled = false; loading = true; error = nil
        task = Task {
            do {
                guard let backend = actions?.hosts.backend(for: entry.hostID) else { throw MoldClientError.malformedResponse }
                let result = try await backend.exportOptions()
                try Task.checkCancellation()
                guard !cancelled else { return }
                options = result
                format = formats.first ?? "gif"
                playback = result.playbackChoices.first ?? .loop
                repeatMode = result.repeatChoices.first ?? .forever
                pauseText = String(pauseControl?.defaultValue ?? 0)
                selectFormat()
                if formats.isEmpty { error = "This machine offers no conversions for this print." }
            } catch { if !cancelled { self.error = error.localizedDescription } }
            loading = false
        }
    }
    func cancel() {
        cancelled = true; task?.cancel(); converting = false
        if actions?.activeExportID == id { actions?.busy = false; actions?.activeExportID = nil }
    }
    func submit() {
        guard valid, let actions, !actions.busy else { return }
        actions.busy = true; actions.activeExportID = id; converting = true; error = nil
        let video = VideoExportRequest(format: format, playback: playback, repeatMode: repeatMode,
            maxDimension: maxDimension == 0 ? nil : maxDimension, fps: fps == 0 ? nil : fps,
            pauseMs: takesPause ? Int(pauseText) : nil)
        let turntable = MeshTurntableOptions(frames: frames, fps: fps, maxDimension: maxDimension,
            transparent: transparent, playback: playback, repeatMode: repeatMode, pauseMs: video.effectivePauseMs)
        let mesh = animation ? MeshExportRequest.turntable(format: format, turntable) : .geometry(format: format, geometry)
        let output = VideoExportRequest.filename(entry.print.filename, format: format)
        let pickedDestination = destination
        task = Task {
            var staged: URL?
            defer {
                if let staged { PrintActions.removeFiles([staged]) }
                converting = false
                if actions.activeExportID == id { actions.busy = false; actions.activeExportID = nil }
            }
            do {
                guard let backend = actions.hosts.backend(for: entry.hostID) else { throw MoldClientError.malformedResponse }
                let data: Data
                if kind == .mesh { data = try await backend.export(entry.print.filename, request: mesh) }
                else { data = try await backend.export(entry.print.filename, request: video) }
                try Task.checkCancellation()
                guard !cancelled else { return }
                let url = try ExportFiles.stage(data, filename: output); staged = url
                if pickedDestination == .photos {
                    let allowed = await requestPhotosAccess()
                    try Task.checkCancellation()
                    guard PhotosAccess.canSave(allowed) else {
                        actions.permissionRecovery = PermissionRecovery.photos(allowed)
                        error = "Allow Photos access to save this GIF."; return
                    }
                    try Task.checkCancellation()
                    try await PhotosWriter.save([.init(url: url, video: false)])
                    try Task.checkCancellation()
                    actions.status = "Saved to Photos."; actions.sheet = nil
                } else if pickedDestination == .folder {
                    let saved = try ExportFiles.saveToFolder(url)
                    actions.status = "Saved to Files ▸ Mold ▸ \(saved.lastPathComponent)."; actions.sheet = nil
                } else {
                    actions.pendingDelivery = pickedDestination == .files ? .files([url]) : .share([url])
                    staged = nil // Pending delivery now owns this operation's file.
                    actions.sheet = nil
                }
                if kind == .mesh && animation { UserDefaults.standard.set(transparent, forKey: "turntable-transparent") }
            } catch { if !cancelled { self.error = error.localizedDescription } }
        }
    }
}

enum ExportDestination: String, CaseIterable, Identifiable {
    case share, folder, files, photos
    var id: Self { self }
    var title: String {
        switch self { case .share: "Share"; case .folder: "Save to Mold folder"; case .files: "Save to Files"; case .photos: "Save to Photos" }
    }
}
