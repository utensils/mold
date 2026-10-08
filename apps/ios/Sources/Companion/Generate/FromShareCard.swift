import MoldClient
import SwiftUI

/// "From Share" (DESIGN.md §5.9): a picture the Share extension left in the
/// App Group, offered at the top of Generate with the use it was shared for
/// first -- Start From, Reference Image, or Add to Library on the chosen
/// machine -- and Discard.
struct FromShareCard: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @Environment(AppRouter.self) private var router
    @Environment(\.scenePhase) private var phase
    @Environment(\.dynamicTypeSize) private var size
    @ScaledMetric(relativeTo: .body) private var side: CGFloat = 56
    @State private var items: [ShareInbox.Item] = []
    @State private var problem: String?
    @State private var working = false

    var body: some View {
        Group {
            if let item = items.first { card(item) }
        }
        .task { reload() }
        .onChange(of: phase) { _, new in if new == .active { reload() } }
        .onChange(of: router.pendingInbox) { _, _ in reload(); router.pendingInbox = nil }
    }

    private func reload() { items = ShareInbox.items() }

    private func card(_ item: ShareInbox.Item) -> some View {
        let caps = generate.recipe?.capabilities
        let references = caps?.referenceImages(family: generate.model?.family, model: generate.modelName)
        let canStart = PictureWells.showsSourceWell(capabilities: caps,
            mode: SourceImageMode(references: references))
        let typed = caps?.generationReferences
        let canType = typed?.mode.isVisible == true && typed?.kinds.contains("image") == true
        let canRefer = references?.hasRoom(for: generate.draft.media.editImages.count) == true
        let stacked = RowAxis.for(size) == .vertical
        return VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                if let image = UIImage(contentsOfFile: ShareInbox.url(of: item).path) {
                    Image(uiImage: image).resizable().scaledToFill()
                        .frame(width: side, height: side).clipShape(.rect(cornerRadius: 10))
                        .accessibilityLabel("The shared picture")
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("From Share").font(.headline)
                    Text(items.count > 1 ? String(localized: "\(items.count) pictures waiting") : String(localized: "A picture is waiting"))
                        .foregroundStyle(.secondaryText)
                    if let problem { Text(problem).foregroundStyle(.secondaryText) }
                }
            }
            let layout = stacked ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8)) : AnyLayout(HStackLayout(spacing: 8))
            layout {
                ForEach(order(item.use), id: \.self) { use in
                    switch use {
                    case .source where canStart:
                        Button("Start From") { attach(item) { DraftPictureAttachment.useAsSource($0, in: &generate.draft, recipe: generate.recipe) } }
                    case .reference where canRefer || canType:
                        Button("Reference") {
                            attach(item) { picked in
                                if let references {
                                    DraftPictureAttachment.addReference(picked, to: &generate.draft, capability: references, recipe: generate.recipe)
                                } else {
                                    var next = generate.draft.media
                                    next.appendGenerationReference(try GenerationReferenceImporter.image(picked))
                                    if let caps, let error = next.generationReferenceError(capabilities: caps, allowIncomplete: true) {
                                        throw MoldClientError.unreachable(error)
                                    }
                                    generate.draft.media = next
                                }
                            }
                        }
                    case .library where generate.target != nil:
                        Button("Add to Library") { importToLibrary(item) }
                    default:
                        EmptyView()
                    }
                }
                if let named = caps?.mesh?.namedViews, named.mode.isVisible {
                    Menu("Use as View") {
                        ForEach(named.roles, id: \.self) { role in
                            Button(role.rawValue.capitalized) {
                                attach(item) { picked in
                                    let reference = try GenerationReferenceImporter.image(picked, role: role)
                                    generate.draft.media.generationReferences.removeAll { $0.role == role }
                                    generate.draft.media.appendGenerationReference(reference)
                                }
                            }
                        }
                    }
                }
                Button("Discard", role: .destructive) { finish(item) }
            }
            .buttonStyle(.bordered)
            .disabled(working)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular, in: .rect(cornerRadius: 16))
        .padding(.horizontal, 16)
        .accessibilityElement(children: .contain)
    }

    /// The use it was shared for, first.
    private func order(_ first: ShareInbox.Use) -> [ShareInbox.Use] {
        [first] + ShareInbox.Use.allCases.filter { $0 != first }
    }

    private var referenceAccepting: Set<String> {
        generate.recipe?.capabilities.referenceImages(family: generate.model?.family, model: generate.modelName)?.acceptingTypes
            ?? PictureImport.identityReadable
    }

    private func attach(_ item: ShareInbox.Item, _ use: @escaping (ImportedPicture) throws -> Void) {
        working = true
        let fence = ReferenceImportFence(model: generate.modelName, host: generate.target?.id,
            recipe: generate.recipe, media: generate.draft.media)
        let accepting = referenceAccepting
        Task {
            defer { working = false }
            do {
                let picture = try await PictureImport.load(ShareInbox.url(of: item), accepting: accepting)
                guard !Task.isCancelled, fence.isCurrent(model: generate.modelName, host: generate.target?.id,
                    recipe: generate.recipe, media: generate.draft.media) else {
                    problem = "Attachments changed while importing. Choose the shared picture again."
                    return
                }
                try use(picture)
                finish(item)
            } catch {
                problem = error.sentence
            }
        }
    }

    private func importToLibrary(_ item: ShareInbox.Item) {
        guard let host = generate.target else { return }
        working = true
        Task {
            defer { working = false }
            guard let data = try? Data(contentsOf: ShareInbox.url(of: item)) else {
                problem = String(localized: "That picture couldn't be read.")
                return
            }
            if await library.importPicture(data, stem: "shared-\(item.id)", taken: item.created, to: host) {
                finish(item)
            }
        }
    }

    private func finish(_ item: ShareInbox.Item) {
        ShareInbox.remove(item)
        problem = nil
        reload()
    }
}
