import MoldClient
import SwiftUI

// The licence a render would need before it can fetch its model. Split from
// `GeneratePane+Run.swift` for size.
//
// The machine's placement answer lists what admission would download first
// and the exact terms still blocking each (`pending_downloads[].licenses`).
// Queued without them, a gated model's auto-pull fails closed in the worker
// and the render dies with a sentence; asked here, the same sheet the Models
// pane shows takes consent on THAT machine and the press runs again.
extension GeneratePane {
    /// Presents the first outstanding licence and answers true -- the press
    /// is held -- or answers false when there is nothing to accept. Only an
    /// answer for the model being submitted counts: a probe still in flight
    /// after a model switch describes the previous one.
    func holdsForLicence(on host: MoldHost, accepted: Set<String>) -> Bool {
        guard let licence = Self.licenceToAsk(
            placement: controller.probe.placement, answeredFor: controller.probe.placementModel,
            submitting: controller.modelName, accepted: accepted)
        else { return false }
        downloads.pendingLicense = DownloadStore.PendingLicense(
            refusal: licence, mismatch: false, host: host.id
        ) {
            startRun(accepted: accepted.union([licence.id]))
        }
        return true
    }

    /// The first term still blocking `submitting`, or nil. Pure, so a test
    /// holds the rule without a view.
    static func licenceToAsk(
        placement: PlacementPreview?, answeredFor: String?, submitting: String?,
        accepted: Set<String>
    ) -> LicenseRefusal? {
        guard let placement, let submitting, answeredFor == submitting else { return nil }
        return placement.outstandingLicenses(excluding: accepted).first
    }
}
