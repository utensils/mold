import AppKit
import Testing

@testable import Mold

/// Whether anything in the app has a caret in it.
///
/// **Fails today**: the answer is assembled per field from
/// `focusedValue(\.editingText)`, and `ShelfNameSheet`'s Name field and
/// `TagNameSheet`'s field publish none -- so typing a space into either fires
/// the Library menu's Quick Look. There is nothing to ask.
@MainActor
struct TextEditingFocusTests {
    /// A private centre, so one test cannot see another's notifications and
    /// nothing here disturbs the app's own observer.
    private func focus() -> (TextEditingFocus, NotificationCenter) {
        let center = NotificationCenter()
        let focus = TextEditingFocus()
        focus.startObserving(center: center)
        return (focus, center)
    }

    /// The field editor posts these whichever field it is serving -- in a
    /// sheet, in the inspector, in the toolbar -- which is the whole point.
    @Test func theFieldEditorBeginningAndEndingIsWhatDecides() async {
        let (focus, center) = focus()
        #expect(!focus.isEditing)

        center.post(name: NSText.didBeginEditingNotification, object: NSTextView())
        await settle(until: { focus.isEditing })
        #expect(focus.isEditing)

        center.post(name: NSText.didEndEditingNotification, object: NSTextView())
        await settle(until: { !focus.isEditing })
        #expect(!focus.isEditing)
    }

    /// A sheet dismissed mid-edit posts no "ended"; the window giving up key
    /// is what says the caret is gone.
    @Test func aWindowLosingKeyEndsTheEdit() async {
        let (focus, center) = focus()
        center.post(name: NSText.didBeginEditingNotification, object: NSTextView())
        await settle(until: { focus.isEditing })

        center.post(name: NSWindow.didResignKeyNotification, object: nil)
        await settle(until: { !focus.isEditing })
        #expect(!focus.isEditing)
    }

    @Test func observingTwiceRegistersOneSetOfObservers() async {
        let (focus, center) = focus()
        focus.startObserving(center: center)

        center.post(name: NSText.didBeginEditingNotification, object: NSTextView())
        await settle(until: { focus.isEditing })
        center.post(name: NSText.didEndEditingNotification, object: NSTextView())
        await settle(until: { !focus.isEditing })

        #expect(!focus.isEditing)
    }
}
