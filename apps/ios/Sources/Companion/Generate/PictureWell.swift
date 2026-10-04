import MoldClient
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// One well: a picture or an empty square, captioned with what it is for;
/// its menu takes a picture from Photos, the camera, Files, the Library or
/// the pasteboard, and removes it.
struct Well: View {
    @Environment(GenerateController.self) private var generate
    @State private var selectionFence: ReferenceImportFence?
    @State private var loadTask: Task<Void, Never>?
    let title: String
    let image: String?
    let side: CGFloat
    let accepting: Set<String>
    var number: Int?
    let set: (ImportedPicture) -> Void
    let clear: () -> Void

    @State private var photo: PhotosPickerItem?
    @State private var showsPhotos = false
    @State private var showsFiles = false
    @State private var showsCamera = false
    @State private var showsLibrary = false
    @State private var problem: String?

    var body: some View {
        VStack(spacing: 4) {
            Menu {
                Button { capture(); showsPhotos = true } label: { Label("Photos", systemImage: "photo.on.rectangle") }
                if UIImagePickerController.isSourceTypeAvailable(.camera) {
                    Button { capture(); showsCamera = true } label: { Label("Take Photo", systemImage: "camera") }
                }
                Button { capture(); showsFiles = true } label: { Label("Files", systemImage: "folder") }
                Button { capture(); showsLibrary = true } label: { Label("Choose from Library…", systemImage: "photo.on.rectangle.angled") }
                if UIPasteboard.general.hasImages {
                    Button { paste() } label: { Label("Paste", systemImage: "doc.on.clipboard") }
                }
                if image != nil {
                    Divider()
                    Button(role: .destructive, action: clear) { Label("Remove", systemImage: "trash") }
                }
            } label: {
                face
            }
            .accessibilityLabel(image == nil ? String(localized: "\(title), empty") : title)
            Text(title).font(.caption).foregroundStyle(.secondaryText).lineLimit(2)
                .multilineTextAlignment(.center).frame(maxWidth: side)
            if let problem { Text(problem).font(.caption2).foregroundStyle(.red).frame(maxWidth: side * 1.6) }
        }
        // iPad: a picture dragged from Photos, Files or the Library grid.
        .dropDestination(for: Data.self) { items, _ in
            guard let data = items.first else { return false }
            take { try await PictureImport.conforming(data, name: "Dropped", accepting: accepting) }
            return true
        }
        .photosPicker(isPresented: $showsPhotos, selection: $photo, matching: .images)
        .fileImporter(isPresented: $showsFiles, allowedContentTypes: [.image]) { result in
            if case let .success(url) = result { take { try await PictureImport.load(url, accepting: accepting) } }
        }
        .sheet(isPresented: $showsCamera) { CameraCapture { data in take { try await PictureImport.conforming(data, name: "Photo.jpg", accepting: accepting) } } }
        .sheet(isPresented: $showsLibrary) {
            LibraryPicker { data, name in take { try await PictureImport.conforming(data, name: name, accepting: accepting) } }
        }
        .onDisappear { loadTask?.cancel() }
        .onChange(of: photo) { _, item in
            guard let item else { return }
            take {
                guard let data = try await item.loadTransferable(type: Data.self) else { throw CancellationError() }
                return try await PictureImport.conforming(data, name: "Photo", accepting: accepting)
            }
            photo = nil
        }
    }

    @ViewBuilder private var face: some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: image == nil ? [5] : []))
                .foregroundStyle(.secondaryText)
            if let image, let data = Data(base64Encoded: image), let picture = UIImage(data: data) {
                Image(uiImage: picture).resizable().scaledToFill()
                    .frame(width: side, height: side).clipShape(.rect(cornerRadius: 10))
            } else {
                // a11y: decorative -- the well's label says it is empty.
                Image(systemName: "plus").font(.title3).foregroundStyle(.secondaryText)
                    .frame(maxWidth: .infinity, maxHeight: .infinity).accessibilityHidden(true)
            }
            if let number, image != nil { Badge(text: "\(number)", mono: true) }
        }
        .frame(width: side, height: side)
    }

    private func paste() {
        guard let data = UIPasteboard.general.image?.pngData() else { return }
        take { try await PictureImport.conforming(data, name: "Pasted.png", accepting: accepting) }
    }

    private var currentFence: ReferenceImportFence {
        ReferenceImportFence(model: generate.modelName, host: generate.target?.id,
            recipe: generate.recipe, media: generate.draft.media)
    }
    private func capture() { selectionFence = currentFence }

    private func take(_ load: @escaping () async throws -> ImportedPicture) {
        problem = nil
        let fence = selectionFence ?? currentFence
        selectionFence = nil
        loadTask?.cancel()
        loadTask = Task {
            do {
                let picture = try await load()
                try Task.checkCancellation()
                guard fence.isCurrent(model: generate.modelName, host: generate.target?.id,
                                      recipe: generate.recipe, media: generate.draft.media) else {
                    problem = "Attachments changed while choosing. Choose the picture again."
                    return
                }
                set(picture)
            } catch is CancellationError {} catch { problem = error.localizedDescription }
        }
    }
}
