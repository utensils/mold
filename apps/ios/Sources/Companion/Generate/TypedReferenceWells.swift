import MoldClient
import SwiftUI
import UniformTypeIdentifiers

/// Ordered multimodal references and semantic camera views use distinct wells.
struct TypedReferenceWells: View {
    @Environment(GenerateController.self) private var generate
    @ScaledMetric(relativeTo: .body) private var side: CGFloat = 72
    @State private var files = false
    @State private var problem: String?
    @State private var importing = false
    @State private var replacement: GenerationReference?
    @State private var replacementIndex: Int?
    @State private var fileFence: ReferenceImportFence?
    @State private var importTask: Task<Void, Never>?

    var body: some View {
        if let caps = generate.recipe?.capabilities {
            VStack(alignment: .leading, spacing: 8) {
                if let named = caps.mesh?.namedViews, named.mode.isVisible {
                    Text("Camera views").font(.headline)
                    ScrollView(.horizontal) {
                        HStack(alignment: .top, spacing: 10) {
                            ForEach(named.roles, id: \.self) { role in
                                let reference = generate.draft.media.generationReferences.first { $0.role == role }
                                Well(title: role.rawValue.capitalized, image: reference?.media.data, side: side,
                                     accepting: PictureImport.identityReadable,
                                     set: { picture in attachImage(picture, role: role) },
                                     clear: { generate.draft.media.generationReferences.removeAll { $0.role == role } })
                            }
                        }
                    }
                } else if let cap = caps.generationReferences, cap.mode.isVisible {
                    Text("References").font(.headline)
                    Text("Use image 1, video 1, or audio 1 in your prompt. Reference order is preserved.")
                        .font(.caption).foregroundStyle(.secondaryText)
                    ScrollView(.horizontal) {
                        HStack(alignment: .top, spacing: 10) {
                            ForEach(Array(generate.draft.media.generationReferences.enumerated()), id: \.offset) { index, reference in
                                referenceWell(reference, index: index)
                            }
                            if generate.draft.media.generationReferences.count < cap.maxCount {
                                if cap.kinds.contains("image"), count("image") < cap.maxImages {
                                    Well(title: "Add image", image: nil, side: side, accepting: PictureImport.identityReadable,
                                         set: { attachImage($0) }, clear: {})
                                }
                                if cap.kinds.contains("video") || cap.kinds.contains("audio") {
                                    Button { openFiles() } label: {
                                        Text("Add reference file…").frame(minWidth: 44, minHeight: 44).contentShape(Rectangle())
                                    }
                                        .disabled(importing).accessibilityIdentifier("add-reference-file")
                                }
                            }
                        }
                    }
                    .accessibilityIdentifier("generation-references")
                    Text("\(generate.draft.media.generationReferences.count) of \(cap.maxCount) references")
                        .font(.caption).foregroundStyle(.secondaryText)
                }
                if importing { ProgressView("Reading reference…") }
                if let problem { Text(problem).foregroundStyle(.secondaryText).accessibilityIdentifier("reference-error") }
            }
            .onDisappear { importTask?.cancel() }
            .fileImporter(isPresented: $files, allowedContentTypes: [.image, .movie, .audio], allowsMultipleSelection: replacement == nil) { result in
                if case let .success(urls) = result { importFiles(urls, capabilities: caps) }
            }
        }
    }

    @ViewBuilder private func referenceWell(_ reference: GenerationReference, index: Int) -> some View {
        VStack(spacing: 4) {
            if reference.kind == "image" {
                Well(title: reference.name, image: reference.media.data, side: side, accepting: PictureImport.identityReadable,
                     number: index + 1, set: { attachImage($0, replacing: index) },
                     clear: { generate.draft.media.removeGenerationReference(at: index) })
            } else {
                Text(reference.name).font(.caption).frame(maxWidth: side * 1.5)
                Text(reference.kind.capitalized).font(.caption).foregroundStyle(.secondaryText)
                Button { openFiles(replacing: reference, at: index) } label: {
                    Text("Replace…").frame(minWidth: 44, minHeight: 44).contentShape(Rectangle())
                }
                    .accessibilityLabel("Replace reference \(index + 1)")
                Button { generate.draft.media.removeGenerationReference(at: index) } label: {
                    Text("Remove").frame(minWidth: 44, minHeight: 44).contentShape(Rectangle())
                }
                    .accessibilityLabel("Remove reference \(index + 1)")
            }
            Menu {
                Button("Move earlier") { generate.draft.media.moveGenerationReference(from: index, to: index - 1) }.disabled(index == 0)
                Button("Move later") { generate.draft.media.moveGenerationReference(from: index, to: index + 1) }
                    .disabled(index + 1 == generate.draft.media.generationReferences.count)
            } label: {
                Text("Order").font(.caption).fixedSize(horizontal: false, vertical: true)
                    .frame(minWidth: 44, minHeight: 44).contentShape(Rectangle())
            }
            .accessibilityLabel("Order reference \(index + 1)")
        }
    }

    private func count(_ kind: String) -> Int { generate.draft.media.generationReferences.filter { $0.kind == kind }.count }
    private func attachImage(_ picture: ImportedPicture, role: GenerationImageReferenceRole? = nil, replacing: Int? = nil) {
        do {
            let reference = try GenerationReferenceImporter.image(picture, role: role)
            var next = generate.draft.media
            if let role {
                next.generationReferences.removeAll { $0.role == role }
                next.appendGenerationReference(reference)
            } else if let replacing { next.replaceGenerationReference(at: replacing, with: reference) }
            else { next.appendGenerationReference(reference) }
            if let caps = generate.recipe?.capabilities,
               let error = next.generationReferenceError(capabilities: caps, allowIncomplete: true) {
                throw MoldClientError.unreachable(error)
            }
            generate.draft.media = next
            problem = nil
        } catch { problem = error.sentence }
    }
    private var currentFence: ReferenceImportFence {
        ReferenceImportFence(model: generate.modelName, host: generate.target?.id,
            recipe: generate.recipe, media: generate.draft.media)
    }
    private func openFiles(replacing reference: GenerationReference? = nil, at index: Int? = nil) {
        replacementIndex = index
        replacement = reference; fileFence = currentFence; files = true
    }
    private func importFiles(_ urls: [URL], capabilities: RecipeCapabilities) {
        importing = true; problem = nil
        let replacing = replacement
        let replacingIndex = replacementIndex
        var fence = fileFence ?? currentFence
        replacement = nil; replacementIndex = nil; fileFence = nil
        importTask?.cancel()
        importTask = Task {
            defer { importing = false }
            do {
                for url in urls {
                    let reference = try await GenerationReferenceImporter.load(url: url)
                    try Task.checkCancellation()
                    guard fence.isCurrent(model: generate.modelName, host: generate.target?.id,
                                          recipe: generate.recipe, media: generate.draft.media) else {
                        throw ReferenceImportError("Attachments changed while choosing. Choose the file again.")
                    }
                    var media = generate.draft.media
                    if let replacing {
                        guard let index = replacingIndex, media.generationReferences.indices.contains(index),
                              media.generationReferences[index] == replacing else { throw CancellationError() }
                        media.replaceGenerationReference(at: index, with: reference)
                    } else { media.appendGenerationReference(reference) }
                    if let error = media.generationReferenceError(capabilities: capabilities, allowIncomplete: true) {
                        throw ReferenceImportError(error)
                    }
                    generate.draft.media = media
                    fence = currentFence
                    if replacing != nil { break }
                }
            } catch is CancellationError {} catch { problem = error.sentence }
        }
    }
}
