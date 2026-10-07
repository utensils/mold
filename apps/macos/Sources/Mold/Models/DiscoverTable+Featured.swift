import MoldClient
import SwiftUI

extension DiscoverTable {
    var featured: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Featured Mold Models").font(.title2.weight(.semibold))
                    Text("Models from this machine’s Mold manifest. Choose a model to install or repair.")
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if capabilities?.canBrowseCatalog == true {
                    Button("Browse Community Catalog") { browseCatalog() }.buttonStyle(.bordered)
                }
            }.padding(12)
            let curated = DiscoverLanding.featured(models.all(on: host.id)).filter { $0.matchesDiscovery(CatalogQuery(text: searchText, includeNSFW: false)) }
            if curated.isEmpty {
                ContentUnavailableView("No featured models available", systemImage: "cube",
                                       description: Text("Refresh this machine’s model list or change your search."))
                Button("Refresh Models") { Task { await models.refresh(on: host.id) } }.padding(12)
            } else {
                List(curated) { model in
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(model.headline).font(.headline)
                            if let tradeOff = model.tradeOff { Text(tradeOff).foregroundStyle(.secondary) }
                            Text(model.family).font(.caption).foregroundStyle(.secondary)
                            if let size = model.sizeGb { Text("\(size.formatted()) GB").font(.caption) }
                            if model.runtimeAvailable == false, let reason = model.runtimeUnavailableReason {
                                Text(reason).font(.callout).foregroundStyle(.secondary)
                            }
                        }
                        Spacer()
                        let progress = downloads.progress(for: model.name, on: host.id)
                        if model.runtimeAvailable != false || model.isReady {
                            ModelStateCell(model: model, progress: progress,
                                           install: { model in Task { await downloads.install(model.name, on: host) } },
                                           cancel: downloads.active[host.id]?.first(where: { $0.value.model == model.name }).map { job in
                                               { Task { await downloads.cancel(jobID: job.key, on: host) } }
                                           })
                        }
                    }.padding(.vertical, 8)
                    .accessibilityIdentifier("featured-model-" + model.name)
                }.listStyle(.inset)
            }
        }
    }
}
