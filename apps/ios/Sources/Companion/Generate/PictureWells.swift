import MoldClient
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// The picture wells (DESIGN.md §5.1): "Start from" when the recipe reads a
/// source picture, then references numbered the way a prompt names them
/// ("image 1", "image 2"). Drawn only when the recipe reads them; a picture a
/// new recipe cannot read is parked by the draft, never dropped. HEIC and
/// WebP are converted on the phone (`PictureImport`), alpha kept.
struct PictureWells: View {
    @Environment(GenerateController.self) private var generate
    @ScaledMetric(relativeTo: .body) private var side: CGFloat = 72

    var body: some View {
        let caps = generate.recipe?.capabilities
        let readsSource = caps?.readsSourceImage == true
        let references = caps?.referenceImages(family: generate.model?.family, model: generate.modelName)
        if readsSource || references != nil {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 10) {
                    if readsSource {
                        Well(title: String(localized: "Start from"), image: generate.draft.media.sourceImage,
                             side: side, accepting: PictureImport.engineReadable,
                             set: { picked in
                                 DraftPictureAttachment.useAsSource(picked, in: &generate.draft, recipe: generate.recipe)
                             },
                             clear: { clearSource() })
                    }
                    if let references {
                        ForEach(Array(generate.draft.media.editImages.enumerated()), id: \.offset) { index, image in
                            Well(title: String(localized: "image \(index + 1)"), image: image, side: side,
                                 accepting: PictureImport.engineReadable, number: index + 1,
                                 set: { picked in generate.draft.media.editImages[index] = picked.encoded },
                                 clear: { generate.draft.media.editImages.remove(at: index) })
                        }
                        if references.hasRoom(for: generate.draft.media.editImages.count) {
                            Well(title: String(localized: "Add image"), image: nil, side: side,
                                 accepting: PictureImport.engineReadable,
                                 set: { picked in
                                     DraftPictureAttachment.addReference(picked, to: &generate.draft,
                                                                         capability: references, recipe: generate.recipe)
                                 },
                                 clear: {})
                        }
                    }
                }
            }
        }
    }

    private func clearSource() {
        generate.draft.media.sourceImage = nil
        generate.draft.media.sourceImageName = nil
        generate.draft.media.sourceImageOriginal = nil
        generate.draft.media.sourceImageOriginalName = nil
        generate.draft.media.sourceImagePixels = nil
    }
}

/// One well: a picture or an empty square, captioned with what it is for;
/// its menu takes a picture from Photos, the camera, Files, the Library or
/// the pasteboard, and removes it.
struct Well: View {
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
                Button { showsPhotos = true } label: { Label("Photos", systemImage: "photo.on.rectangle") }
                if UIImagePickerController.isSourceTypeAvailable(.camera) {
                    Button { showsCamera = true } label: { Label("Take Photo", systemImage: "camera") }
                }
                Button { showsFiles = true } label: { Label("Files", systemImage: "folder") }
                Button { showsLibrary = true } label: { Label("Choose from Library…", systemImage: "photo.on.rectangle.angled") }
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
        .photosPicker(isPresented: $showsPhotos, selection: $photo, matching: .images)
        .fileImporter(isPresented: $showsFiles, allowedContentTypes: [.image]) { result in
            if case let .success(url) = result { take { try await PictureImport.load(url, accepting: accepting) } }
        }
        .sheet(isPresented: $showsCamera) { CameraCapture { data in take { try await PictureImport.conforming(data, name: "Photo.jpg", accepting: accepting) } } }
        .sheet(isPresented: $showsLibrary) {
            LibraryPicker { data, name in take { try await PictureImport.conforming(data, name: name, accepting: accepting) } }
        }
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

    private func take(_ load: @escaping () async throws -> ImportedPicture) {
        problem = nil
        Task {
            do { set(try await load()) } catch is CancellationError {} catch {
                problem = error.localizedDescription
            }
        }
    }
}
