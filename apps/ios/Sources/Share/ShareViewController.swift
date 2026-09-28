import ImageIO
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// Share ▸ Mold Studio (DESIGN.md §5.9): the picture, what it is for, and
/// Save -- which stages it in the App Group for the app. No network here.
final class ShareViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        let provider = (extensionContext?.inputItems as? [NSExtensionItem])?
            .flatMap { $0.attachments ?? [] }
            .first { $0.hasItemConformingToTypeIdentifier(UTType.image.identifier) }
        let host = UIHostingController(rootView: ShareSheet(provider: provider) { [weak self] in
            self?.extensionContext?.completeRequest(returningItems: nil)
        })
        addChild(host)
        host.view.frame = view.bounds
        host.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(host.view)
        host.didMove(toParent: self)
    }
}

struct ShareSheet: View {
    let provider: NSItemProvider?
    let done: () -> Void
    @State private var preview: UIImage?
    @State private var file: URL?
    @State private var use: ShareInbox.Use = .source
    @State private var saved = false
    @State private var problem: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    if let preview {
                        Image(uiImage: preview).resizable().scaledToFit().frame(maxHeight: 280)
                            .frame(maxWidth: .infinity)
                            .accessibilityLabel("The shared picture")
                    } else if problem == nil {
                        ProgressView().frame(maxWidth: .infinity)
                    }
                    if let problem { Text(problem) }
                }
                if saved {
                    Section {
                        Label("Waiting in Mold Studio. Open it to finish.", systemImage: "checkmark.circle")
                    }
                } else {
                    Section("Use As") {
                        Picker("Use As", selection: $use) {
                            Text("Start From").tag(ShareInbox.Use.source)
                            Text("Reference Image").tag(ShareInbox.Use.reference)
                            Text("Add to Library").tag(ShareInbox.Use.library)
                        }
                        .pickerStyle(.inline)
                        .labelsHidden()
                    }
                }
            }
            .navigationTitle("Mold Studio")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if saved {
                    ToolbarItem(placement: .confirmationAction) { Button("Done", action: done) }
                } else {
                    ToolbarItem(placement: .cancellationAction) { Button("Cancel", action: done) }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Save", action: save).disabled(file == nil)
                    }
                }
            }
        }
        .task { await load() }
    }

    /// The file iOS hands over, kept for Save, and a screen-sized preview
    /// read from it without decoding the whole picture.
    private func load() async {
        guard let provider else { problem = String(localized: "There's no picture to share."); return }
        let copy: URL? = await withCheckedContinuation { done in
            _ = provider.loadFileRepresentation(for: .image) { url, _, _ in
                guard let url else { return done.resume(returning: nil) }
                let copy = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString + "-" + url.lastPathComponent)
                done.resume(returning: (try? FileManager.default.copyItem(at: url, to: copy)) != nil ? copy : nil)
            }
        }
        guard let copy else { problem = String(localized: "That picture couldn't be read."); return }
        file = copy
        if let source = CGImageSourceCreateWithURL(copy as CFURL, nil),
           let thumb = CGImageSourceCreateThumbnailAtIndex(source, 0, [
               kCGImageSourceCreateThumbnailFromImageAlways: true,
               kCGImageSourceCreateThumbnailWithTransform: true,
               kCGImageSourceThumbnailMaxPixelSize: 800,
           ] as CFDictionary) {
            preview = UIImage(cgImage: thumb)
        }
    }

    private func save() {
        guard let file else { return }
        do {
            _ = try ShareInbox.stage(file, use: use)
            try? FileManager.default.removeItem(at: file)
            saved = true
        } catch {
            problem = String(localized: "Mold Studio couldn't keep that picture. Try again.")
        }
    }
}
