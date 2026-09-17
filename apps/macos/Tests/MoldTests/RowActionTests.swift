import Foundation
import MoldClient
import Testing

@testable import Mold

/// Every list row carries its actions on a right-click, built from the same
/// declaration its inline controls use. Pure, so the menus are pinned without
/// rendering one -- `MachineSelection.rows`' own idiom for the Machine menu.
@MainActor
struct RowActionTests {
    // MARK: - The shared ordering rule

    /// Whatever order a surface declares them in, everything destructive ends
    /// up last, so a menu's bottom item is always the one that cannot be
    /// taken back.
    @Test func destructiveActionsSortToTheEnd() {
        let declared = [
            RowAction(kind: "remove", title: "Remove…", isDestructive: true),
            RowAction(kind: "edit", title: "Edit…"),
            RowAction(kind: "check", title: "Check Now"),
        ]
        #expect(RowAction.ordered(declared).map(\.kind) == ["edit", "check", "remove"])
    }

    /// **Fails today**: `ProviderSection` attached `.contextMenu` regardless,
    /// so a right-click on a provider with no token stored opened an empty
    /// menu -- which says there is something here and then does not say what.
    /// A disabled placeholder is the same lie with an extra row. The rule
    /// belongs to the shared type, because every caller has the case.
    @Test func aRowWithNoApplicableActionCarriesNoMenu() {
        #expect(!RowAction<String>.offersMenu([]))
        #expect(RowAction.offersMenu([RowAction(kind: "clear", title: "Clear")]))

        // Accounts is the surface that has the empty case.
        #expect(AccountsRow.unset.menu(named: "Civitai").isEmpty)
        #expect(AccountsRow.environment(masked: "hf_••••").menu(named: "Hugging Face").isEmpty,
                "an environment token has nothing on this machine to clear")
        let stored = AccountsRow.stored(masked: "hf_••••").menu(named: "Hugging Face")
        #expect(stored.map(\.title) == ["Clear Hugging Face Token"])
        #expect(stored.map(\.isDestructive) == [true])
    }

    /// The sweep the other call sites need: every shape they can hand
    /// `.rowActionMenu` resolves to at least one action, so none of them can
    /// reach the empty case by accident.
    @Test func everyOtherSurfaceAlwaysHasSomethingToOffer() {
        for isManaged in [true, false] {
            for isDefault in [true, false] {
                #expect(RowAction.offersMenu(
                    MachineRowActions.offered(isManaged: isManaged, isDefault: isDefault)))
            }
        }

        // An env-locked row cannot be written or reset; a null secret cannot
        // be copied. Copy Key is what keeps both from being empty.
        let awkward = [
            FakeFixtures.configEntry("default_width", value: .number(768), source: "env",
                                     envVar: "MOLD_DEFAULT_WIDTH"),
            FakeFixtures.configEntry("default_width", value: .number(768), source: "env"),
            FakeFixtures.configEntry("runpod.api_key", value: .null, source: "default"),
            FakeFixtures.configEntry("models_dir", value: .string(""), source: "file"),
        ]
        for entry in awkward {
            let advanced = ConfigRowActions.offered(for: entry)
            #expect(RowAction.offersMenu(advanced), "Advanced: \(entry.key)/\(entry.source)")
            // A curated pane filters Reset out -- it has no reset control to
            // mirror -- and must still be left with something.
            #expect(RowAction.offersMenu(advanced.filter { $0.kind != .reset }),
                    "curated: \(entry.key)/\(entry.source)")
        }
    }

    // MARK: - Settings ▸ Machines

    @Test func aMachineRowOffersEverythingTheFooterDoes() {
        let actions = MachineRowActions.offered(isManaged: true, isDefault: false)

        #expect(actions.map(\.kind) == [.edit, .checkNow, .copyAddress, .setDefault, .remove])
        #expect(actions.last?.isDestructive == true)
        #expect(actions.allSatisfy { !$0.isDisabled })
    }

    /// This Mac's engine is not a saved row: there is nothing to edit and
    /// nothing to remove. Both stay PRESENT and disabled -- a menu that
    /// changes shape per row is one nobody learns.
    @Test func theLocalEnginesRowCannotBeEditedOrRemoved() {
        let actions = MachineRowActions.offered(isManaged: false, isDefault: true)
        let disabled = actions.filter(\.isDisabled).map(\.kind)

        #expect(disabled == [.edit, .setDefault, .remove])
        #expect(actions.count == 5)
    }

    // MARK: - Config rows

    @Test func aConfigRowCopiesItsKeyItsValueAndItsVariable() {
        let entry = FakeFixtures.configEntry(
            "models_dir", value: .string("/data/models"), source: "env", envVar: "MOLD_MODELS_DIR")
        let actions = ConfigRowActions.offered(for: entry)

        #expect(actions.map(\.kind) == [.copyKey, .copyValue, .copyVariable])
        #expect(ConfigRowActions.copied(.copyKey, from: entry) == "models_dir")
        #expect(ConfigRowActions.copied(.copyValue, from: entry) == "/data/models")
        #expect(ConfigRowActions.copied(.copyVariable, from: entry) == "MOLD_MODELS_DIR")
    }

    /// Reset is offered on exactly `canReset`'s own gate -- `source == "db"`,
    /// which is DELETE's gate on the machine -- and it is the last item,
    /// behind the divider.
    @Test func resetIsOfferedWhereTheMachineWouldAcceptIt() {
        let sources = ["db", "file", "env", "default"]
        let offered = sources.map { source in
            ConfigRowActions.offered(
                for: FakeFixtures.configEntry("gallery.trash_retention_days",
                                              value: .number(30), source: source))
        }

        #expect(offered.map { $0.contains { $0.kind == .reset } } == [true, false, false, false])
        #expect(offered[0].last?.kind == .reset)
        #expect(offered[0].last?.isDestructive == true)
    }

    /// A menu item that put a machine's API key on the pasteboard because
    /// somebody right-clicked near it is not a convenience. The field itself
    /// only ever shows a mask.
    @Test func aSecretsValueIsNeverCopyable() {
        let secret = FakeFixtures.configEntry("runpod.api_key", value: .string("<set>"), source: "db")

        #expect(!ConfigRowActions.offered(for: secret).contains { $0.kind == .copyValue })
        #expect(ConfigRowActions.copied(.copyValue, from: secret) == nil)
    }

    /// An unset row has no value to take anywhere.
    @Test func anUnsetRowOffersNoCopyValue() {
        let unset = FakeFixtures.configEntry("lambda.api_key", value: .null, source: "default")
        let empty = FakeFixtures.configEntry("models_dir", value: .string(""), source: "db")

        #expect(!ConfigRowActions.offered(for: unset).contains { $0.kind == .copyValue })
        #expect(!ConfigRowActions.offered(for: empty).contains { $0.kind == .copyValue })
    }
}
