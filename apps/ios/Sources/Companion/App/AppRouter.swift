import MoldClient
import Observation

/// Where the app is and which sheet is up, so a button on one tab, a menu
/// command or a deep link can send it somewhere else.
@Observable
final class AppRouter {
    var selection: TabSelection = .go(.generate)
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

    /// "Add a Machine…" from anywhere: Machines, with the sheet up.
    func addMachine(_ prefill: AddPrefill? = nil) {
        selection = .go(.machines)
        addPrefill = prefill
        showsAddMachine = true
    }
}
