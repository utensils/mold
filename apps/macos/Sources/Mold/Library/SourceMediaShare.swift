import AppKit
import MoldClient

/// The private original belongs to the picker and its selected sharing service.
@MainActor
final class SourceMediaShare: NSObject, NSSharingServicePickerDelegate, NSSharingServiceDelegate {
    private(set) var file: SourceMediaFile?
    private var picker: NSSharingServicePicker?
    private var service: NSSharingService?
    private var isCurrent: (() -> Bool)?
    func present(_ file: SourceMediaFile, relativeTo anchor: NSView, isCurrent: @escaping () -> Bool) {
        clear()
        guard isCurrent() else { return }
        self.file = file; self.isCurrent = isCurrent
        let picker = NSSharingServicePicker(items: [file.url])
        picker.delegate = self; self.picker = picker
        picker.show(relativeTo: NSRect(x: anchor.bounds.midX, y: anchor.bounds.midY, width: 1, height: 1), of: anchor, preferredEdge: .minY)
    }
    func clear() {
        picker?.delegate = nil; service?.delegate = nil
        picker = nil; service = nil; file = nil; isCurrent = nil
    }
    func sharingServicePicker(_ sharingServicePicker: NSSharingServicePicker, didChoose service: NSSharingService?) {
        guard let service, isCurrent?() == true else { clear(); return }
        self.service = service; service.delegate = self
        picker = nil
    }
    func sharingService(_ sharingService: NSSharingService, didShareItems items: [Any]) { clear() }
    func sharingService(_ sharingService: NSSharingService, didFailToShareItems items: [Any], error: any Error) { clear() }
}
