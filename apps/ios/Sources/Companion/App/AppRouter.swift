import Foundation
import MoldClient
import Observation

/// Where the app is and which sheet is up, so a button on one tab, a menu
/// command or a deep link can send it somewhere else.
@Observable
final class AppRouter {
    let presentationID = UUID()
    var licenseDetailContext: (host: UUID, job: String)?
    var selection: TabSelection = .go(.generate)
    var isInLibrary: Bool {
        switch selection {
        case .go(.library), .search, .shelf: true
        default: false
        }
    }
    var showsSettings = false
    var showsAddMachine = false
    /// What the Add sheet opens with: a Nearby machine's resolved address,
    /// or (from M3) a scanned pairing code.
    var addPrefill: AddPrefill?

    struct AddPrefill: Equatable {
        var name: String
        var address: String
    }

    /// A print whose recipe Generate should load next ("Use These Settings").
    var pendingReuse: LibraryEntry?
    var pendingSource: LibraryEntry?

    func useAsSource(_ entry: LibraryEntry) {
        pendingSource = entry
        selection = .go(.generate)
    }

    func reuse(_ entry: LibraryEntry) {
        pendingReuse = entry
        selection = .go(.generate)
    }

    /// A print opened by a link (widget, notification), over whatever is up.
    var openedPrint: OpenedPrint?
    /// A Share-extension photo Generate should offer next.
    var pendingInbox: String?

    struct OpenedPrint: Identifiable, Equatable { let id: PrintID }

    /// Where a `moldstudio://` link goes.
    func open(_ link: DeepLink) {
        switch link {
        case let .print(host, filename):
            openedPrint = OpenedPrint(id: PrintID(host: host, filename: filename))
        case .queue:
            selection = .go(.queue)
        case let .generate(inbox):
            pendingInbox = inbox
            selection = .go(.generate)
        }
    }

    /// A pairing code opened from outside (the Camera app reading the Mac's
    /// QR, a tapped `https://utensils.io/mold/pair#…`), waiting for the
    /// person to say Pair -- never claimed on its own.
    var pairingLink: PairingLink?

    struct PairingLink: Identifiable, Equatable {
        let id = UUID()
        /// `nil` when the code could not be read; `failure` says why.
        let payload: MobilePairingPayload?
        let failure: String?
    }

    /// Any URL the system hands the app: a pairing link, or a
    /// `moldstudio://` link. Anything else is ignored.
    func open(url: URL) {
        if MobilePairingPayload.isPairingLink(url) {
            // UIKit presents nothing over a sheet already up: close it.
            showsAddMachine = false
            showsSettings = false
            openedPrint = nil
            selection = .go(.machines)
            do {
                pairingLink = PairingLink(payload: try MobilePairingPayload.parse(url.absoluteString), failure: nil)
            } catch {
                pairingLink = PairingLink(payload: nil, failure: error.errorDescription)
            }
        } else if let link = DeepLink(url) {
            open(link)
        }
    }

    /// "Add a Machine…" from anywhere: Machines, with the sheet up.
    func addMachine(_ prefill: AddPrefill? = nil) {
        selection = .go(.machines)
        addPrefill = prefill
        showsAddMachine = true
    }
}
