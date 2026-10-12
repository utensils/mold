import MoldClient
import SwiftUI

struct SourceMediaModelChooser: View {
    @Environment(HostStore.self) private var hosts
    @Environment(ModelStore.self) private var models
    @Environment(GenerateController.self) private var generate
    @Environment(\.dismiss) private var dismiss
    @State private var search = ""
    let member: RetainedSourceMedia.Member
    let select: (MoldHost, Model, GenerationRecipe, SourceMediaAttachment) -> Void

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text("Choose a model and input role for \(member.displayName). Existing settings are fitted to the chosen recipe.")
                    .foregroundStyle(.secondary)
                List {
                    ForEach(hosts.hosts.filter(hosts.isUp)) { host in
                        Section(host.name) {
                            ForEach(FuzzyModelSearch.models(models.ready(on: host.id), query: search)) { model in
                                let offers = SourceMediaReuseOffer.offers(model: model, member: member, draft: generate.draft)
                                ForEach(offers) { offer in
                                    HStack {
                                        VStack(alignment: .leading) {
                                            Text(model.name).lineLimit(1).truncationMode(.middle)
                                            Text(offer.recipe.label).font(.caption).foregroundStyle(.secondary)
                                        }
                                        Spacer()
                                        Menu("Use as…") {
                                            ForEach(offer.choices) { role in
                                                Button(role.title) { select(host, model, offer.recipe, role); dismiss() }
                                            }
                                        }.fixedSize()
                                    }
                                }
                            }
                        }
                    }
                }
                if models.isLoading { ProgressView("Checking models…") }
                Text("Only installed compatible models on connected machines are shown.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding()
            .searchable(text: $search, prompt: "Search models")
            .navigationTitle("Use source")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem { Button("Refresh", systemImage: "arrow.clockwise") { Task { await models.refresh() } }.disabled(models.isLoading) }
            }
        }
        .frame(minWidth: 560, idealWidth: 640, minHeight: 400, idealHeight: 480)
        .task { await models.refresh() }
        .onExitCommand { dismiss() }
    }
}
