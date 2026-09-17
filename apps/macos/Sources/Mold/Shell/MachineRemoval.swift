import MoldClient

/// The question asked before a machine goes.
///
/// One press of a borderless, icon-only "−" sitting right beside "+" used to
/// remove the machine AND irreversibly destroy the API key stored for it, with
/// no dialog and no undo (review 05-H6). The web counterpart has always asked,
/// naming exactly this consequence (`web/src/pages/MachinesPage.vue:203-217`),
/// and this app already had the primitive: `Destruction` plus
/// `.destructionDialog`, used for the far less destructive preferences reset.
///
/// Never a typed phrase -- `DestructionDialog.swift`'s own rule: making
/// somebody retype a word does not make them read the sentence.
enum MachineRemoval {
    static func destruction(of host: MoldHost, perform: @escaping () -> Void) -> Destruction {
        Destruction(title: "Remove “\(host.name)”?", message: message(for: host),
                    verb: "Remove", perform: perform)
    }

    /// The key clause is stated only where there IS a key: telling somebody a
    /// credential will be forgotten when none was ever stored is the kind of
    /// sentence that stops being read.
    static func message(for host: MoldHost) -> String {
        let address = HostAddress.displayString(for: host.baseURL)
        let key = host.apiKey == nil
            ? ""
            : " The API key stored for it on this Mac is removed with it."
        return "Mold stops talking to \(address) and forgets it.\(key) "
            + "Nothing on the machine itself changes, and you can add it again."
    }
}
