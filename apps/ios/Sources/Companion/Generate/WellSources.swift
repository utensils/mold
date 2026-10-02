import MoldClient
import SwiftUI
import UIKit

/// The camera, for a photo to start from (UIKit's picker: the system's own
/// capture screen, with its own accessibility).
struct CameraCapture: UIViewControllerRepresentable {
    @Environment(\.dismiss) private var dismiss
    let taken: (Data) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ controller: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraCapture
        init(parent: CameraCapture) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage, let data = image.jpegData(compressionQuality: 0.92) {
                parent.taken(data)
            }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { parent.dismiss() }
    }
}

/// "Choose from Library…": the Library's stills, and the chosen print's own
/// bytes fetched from the machine that holds it.
struct LibraryPicker: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    @ScaledMetric(relativeTo: .body) private var tile: CGFloat = 96
    let picked: (Data, String) -> Void

    var body: some View {
        NavigationStack {
            let stills = library.pool.filter { $0.print.kind == .picture }
            Group {
                if stills.isEmpty {
                    EmptyState(title: String(localized: "No pictures yet"), symbol: "photo",
                               message: String(localized: "Pictures you generate appear here to start from."))
                } else {
                    ScrollView {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: tile), spacing: 3)], spacing: 3) {
                            ForEach(stills) { entry in
                                Button { choose(entry) } label: {
                                    Color.clear.aspectRatio(1, contentMode: .fit)
                                        .overlay { PrintThumbnail(entry: entry, points: tile * 1.5) }
                                        .clipShape(.rect(cornerRadius: 5))
                                }
                                .buttonStyle(.plain)
                                .accessibilityLabel(entry.spokenDescription(showsHost: false))
                            }
                        }
                    }
                }
            }
            .navigationTitle("Choose from Library")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } } }
        }
    }

    private func choose(_ entry: LibraryEntry) {
        // A merged tile can lead with a saved copy on an offline machine.
        // Prefer the same print on a machine that is currently answering.
        let source = entry.presented(onAnyOf: Set(hosts.upHosts.map(\.id))) ?? entry
        guard let host = hosts.host(source.hostID) else { return }
        Task {
            do {
                let data = try await hosts.backend(for: host).media(source.print.filename, trashed: false)
                picked(data, source.print.filename)
                dismiss()
            } catch {
                hosts.report(host, doing: String(localized: "fetch that picture"), error)
            }
        }
    }
}
