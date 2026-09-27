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
    /// What the licence gate knows about the render being submitted.
    enum LicenceGate: Equatable {
        /// These terms must be accepted first.
        case ask(LicenseRefusal)
        /// A current answer for this machine and model needs nothing.
        case clear
        /// No answer for this machine and model yet -- the probe is debounced,
        /// or it last answered for another model or machine. Never read as
        /// clear: Generate asks the machine before submitting.
        case unknown
    }

    /// Settles the gate for this press. `.ask` presents the sheet (accepting
    /// runs the press again); `.unknown` asks the machine now, then runs the
    /// press again -- so a gated model never reaches the worker unasked just
    /// because Generate beat the debounce. Answers true when the press is held.
    func holdsForLicence(on host: MoldHost, accepted: Set<String>) -> Bool {
        let probe = controller.probe
        switch Self.licenceGate(
            placement: probe.placement, answeredFor: probe.placementModel,
            answeredOn: probe.placementHost, submitting: controller.modelName, on: host.id,
            accepted: accepted)
        {
        case .clear:
            return false
        case .ask(let licence):
            downloads.pendingLicense = DownloadStore.PendingLicense(
                refusal: licence, mismatch: false, host: host.id
            ) {
                startRun(accepted: accepted.union([licence.id]))
            }
            return true
        case .unknown:
            let pressed = LicenceSubmission(
                draft: controller.draft, model: controller.modelName, host: host.id)
            Task {
                let answered = await probe.settle(
                    draft: pressed.draft, model: pressed.model, on: host, hosts: hosts)
                // The press resumes only if nothing moved while the machine
                // answered; otherwise the new selection waits for its own press
                // and its own gate.
                guard Self.resumes(pressed, now: currentSubmission) else { return }
                // A machine that cannot answer (an older host, a dropped
                // connection) is not a reason to refuse THIS render: the worker
                // still fails closed on a gated fetch, exactly as before.
                startRun(accepted: accepted, licenceSettled: !answered)
            }
            return true
        }
    }

    /// What a Generate press submits, for telling whether it is still the
    /// render on screen after the licence probe's await.
    struct LicenceSubmission: Equatable {
        let draft: RenderDraft
        let model: String?
        let host: MoldHost.ID
    }

    private var currentSubmission: LicenceSubmission? {
        host.map { LicenceSubmission(draft: controller.draft, model: controller.modelName, host: $0.id) }
    }

    /// Whether the continuation may run the press it was started for.
    static func resumes(_ pressed: LicenceSubmission, now: LicenceSubmission?) -> Bool {
        now == pressed
    }

    /// The gate for `submitting` on `host`. Pure, so a test holds the rule
    /// without a view.
    static func licenceGate(
        placement: PlacementPreview?, answeredFor: String?, answeredOn: MoldHost.ID?,
        submitting: String?, on host: MoldHost.ID, accepted: Set<String>
    ) -> LicenceGate {
        guard let placement, let submitting, answeredFor == submitting, answeredOn == host else {
            return .unknown
        }
        return placement.outstandingLicenses(excluding: accepted).first.map(LicenceGate.ask) ?? .clear
    }
}
