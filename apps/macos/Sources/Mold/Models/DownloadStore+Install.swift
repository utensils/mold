import Foundation
import MoldClient

// Starting a fetch, whichever door its name opens, and the licence-refusal
// recovery that follows the one failure either door can answer.
extension DownloadStore {
    /// Asks a machine to fetch a model, by whichever door its NAME opens.
    ///
    /// A manifest name goes to `POST /api/downloads`, which takes a body and
    /// answers a job id (409 included -- already a success, the click's
    /// wanted outcome). A `cv:`/`hf:` id goes to
    /// `POST /api/catalog/:id/download`, which takes no body and answers 202
    /// with a primary job id that may be null when only companions were
    /// missing -- also not a failure. Each route refuses the other with a
    /// generic 400, so the shape of the NAME decides, never the screen that
    /// called (design fact 2 / decision 4, M5).
    ///
    /// A licence refusal from either door is held on `pendingLicense` rather
    /// than reported: it is the one failure a sheet can resolve, by
    /// accepting and running this SAME call again.
    func install(_ name: String, on host: MoldHost) async {
        let client = hosts.backend(for: host)
        do {
            if Model.isCatalogName(name) {
                let result = try await client.installCatalogEntry(id: name)
                track(result.jobIDs, model: name, on: host.id)
            } else {
                let ticket = try await client.startDownload(DownloadRequest(model: name))
                track([ticket.id], model: name, on: host.id)
            }
            hosts.succeeded(on: host.id, doing: "start that download")
            reconcile()
        } catch let MoldClientError.licenseRequired(refusal, mismatch) {
            pendingLicense = PendingLicense(refusal: refusal, mismatch: mismatch, host: host.id) { [weak self] in
                await self?.install(name, on: host)
            }
        } catch {
            hosts.report(error, on: host.id, doing: "start that download")
        }
    }

    /// A licence sheet accepted its terms -- record consent on that ONE
    /// machine, then retry the identical install. Left in place on a failed
    /// acceptance, so the sheet stays up to try again.
    func accepted(_ pending: PendingLicense) async {
        guard await licenses.accept(pending.refusal, on: pending.host) else { return }
        pendingLicense = nil
        await pending.retry()
    }

    /// What the machine is doing, whoever asked. Seeds `active` from
    /// `GET /api/downloads` so a `mold pull` at a terminal, or the web app on
    /// the same box, shows up here even though this store never started it --
    /// without this call, `active` only ever held this app's own clicks.
    func refresh(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            let listing = try await client.downloads()
            adopt(listing, on: host)
            hosts.succeeded(on: host, doing: "list its downloads")
            reconcile()
        } catch {
            hosts.report(error, on: host, doing: "list its downloads")
        }
    }

    private func track(_ jobIDs: [String], model: String, on host: MoldHost.ID) {
        var forHost = active[host] ?? [:]
        for id in jobIDs { forHost[id] = Progress(model: model) }
        active[host] = forHost
    }
}
