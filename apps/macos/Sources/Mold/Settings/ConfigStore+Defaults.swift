import Foundation
import MoldClient

// M3's per-model defaults API, unchanged by the M7 S2 rename -- `defaults`,
// `save` and `clear` keep their exact signatures, so `GenerateController` and
// the inspector's "Use as default for this model" compile and behave exactly
// as they did against `ModelDefaultsStore`.
extension ConfigStore {
    /// A model nobody has ever configured on this host, or whose listing
    /// hasn't been read yet, has no defaults -- `ModelDefaults()` is empty,
    /// so adopting it changes nothing and the recipe's own numbers stand.
    func defaults(for model: String, on host: MoldHost.ID) -> ModelDefaults {
        guard let listing = byHost[host] else { return ModelDefaults() }
        return ModelDefaults(from: listing, model: model)
    }

    /// Writes the fields this draft has a control for, one PUT each, and
    /// re-reads the listing afterwards so what is shown is what the machine
    /// stored rather than what was sent. A partial failure reports once for
    /// this machine -- a retry replaces it the same way every other store's
    /// failure does.
    func save(_ draft: RenderDraft, for model: String, on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        var reported = false
        for (key, value) in ModelDefaults().writes(for: draft, model: model) {
            do {
                try await client.setConfig(key, to: value)
            } catch {
                if !reported {
                    hosts.report(error, on: host, doing: "save the defaults for \(model)")
                    reported = true
                }
            }
        }
        await refresh(on: host)
    }

    /// Drops every one of the eight rows for this model, then re-reads.
    func clear(for model: String, on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        var reported = false
        for key in ModelDefaults.keys(for: model) {
            do {
                try await client.resetConfig(key)
            } catch {
                if !reported {
                    hosts.report(error, on: host, doing: "clear the defaults for \(model)")
                    reported = true
                }
            }
        }
        await refresh(on: host)
    }
}
