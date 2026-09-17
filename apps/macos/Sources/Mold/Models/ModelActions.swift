import MoldClient
import SwiftUI

/// The one door for what can be done to an installed model -- the same call
/// whether it came from the row's contextual menu, the Model menu, or (for
/// Delete) the keyboard. Mirrors `LibraryActions`' shape (M5 S5). The menu
/// contents themselves -- the pure item list and the dispatcher both menus
/// share -- are `ModelActions+Menu.swift`, split for size.
@MainActor
struct ModelActions {
    let hosts: HostStore
    let models: ModelStore
    let downloads: DownloadStore
    let licenses: LicenseStore

    /// Set by the pane, which owns the dialog -- the same reason
    /// `LibraryActions.confirmDestruction` is a closure and not a store
    /// method (decision 12, M5).
    var confirmDestruction: ((Destruction) -> Void)?
    /// Set by the pane, which owns the sheet.
    var presentComponents: ((Model) -> Void)?
    var presentLicense: ((ThirdPartyLicense) -> Void)?
    /// Set by the pane, which owns the transient caption under the table.
    /// There is no `HostFailure`-shaped funnel for a SUCCESS (design S5): a
    /// removal is reported as one sentence here rather than invented as a
    /// new store-wide mechanism for the one caller that needs it.
    var onRemoved: ((String) -> Void)?

    // MARK: - Install / Repair / Cancel

    /// Same route either way -- a repair is a fresh `install` of the same
    /// name, which resumes rather than restarting (`ModelStateCell`'s own
    /// comment on `needsRepair`).
    func install(_ model: Model, on host: MoldHost) {
        Task { await downloads.install(model.name, on: host) }
    }

    func repair(_ model: Model, on host: MoldHost) {
        install(model, on: host)
    }

    /// `nil` off a model that is not mid-download -- there is nothing to
    /// cancel, so no button or menu item is drawn rather than a disabled
    /// one. The one place both the job id and its progress are read from
    /// the same dictionary together (mirrors the former `ModelsPane.cancel`).
    func cancelHandler(for model: Model, on host: MoldHost) -> (() -> Void)? {
        guard let job = downloads.active[host.id]?.first(where: { $0.value.model == model.name })
        else { return nil }
        return { Task { await downloads.cancel(jobID: job.key, on: host) } }
    }

    func isDownloading(_ model: Model, on host: MoldHost.ID) -> Bool {
        downloads.isBusy(model.name, on: host)
    }

    // MARK: - Load / Unload

    func load(_ model: Model, on host: MoldHost.ID) {
        Task { await models.load(model, gpu: nil, on: host) }
    }

    func unload(_ model: Model, on host: MoldHost.ID) {
        Task { await models.unload(model, on: host) }
    }

    // MARK: - Components

    func showComponents(_ model: Model) {
        presentComponents?(model)
    }

    // MARK: - Licence

    func showLicence(_ model: Model, on host: MoldHost.ID) {
        guard let licence = licenses.licence(gating: model.name, on: host) else { return }
        presentLicense?(licence)
    }

    // MARK: - Delete

    /// Asks first. The fake -- and the real machine -- record nothing until
    /// the dialog's own verb is pressed (decision 12/13, M5).
    func delete(_ model: Model, on host: MoldHost.ID) {
        guard let ask = confirmDestruction else { return }
        let size = model.diskUsageBytes.map { Int64($0).formatted(.byteCount(style: .file)) }
        ask(Destruction(
            title: "Delete \(model.headline)?",
            message: size.map { "This frees \($0). Files another installed model still uses are kept." }
                ?? "Files another installed model still uses are kept.",
            verb: "Delete"
        ) {
            Task { await performDelete(model, on: host) }
        })
    }

    private func performDelete(_ model: Model, on host: MoldHost.ID) async {
        guard let removal = await models.delete(model, on: host) else { return }
        onRemoved?(Self.removalSummary(headline: model.headline, removal: removal))
    }

    /// "Removed FLUX.1 Dev Q4 and freed 11.9 GB." -- and, when the delete
    /// kept shared files another model still names, a second sentence
    /// saying how many and for whom (`KeptComponent.usedBy`, design decision
    /// 10/11 and the S5 removal-summary note). Pure so a test can ask the
    /// exact sentence without a store.
    static func removalSummary(headline: String, removal: ModelRemoval) -> String {
        let freed = Int64(removal.freedBytes).formatted(.byteCount(style: .file))
        let base = "Removed \(headline) and freed \(freed)."
        guard !removal.kept.isEmpty else { return base }
        let names = Set(removal.kept.flatMap(\.usedBy)).sorted().joined(separator: ", ")
        let noun = removal.kept.count == 1 ? "file" : "files"
        return "\(base) \(removal.kept.count) shared \(noun) kept for \(names)."
    }
}
