import AppKit

/// Who an UNMODIFIED arrow key belongs to.
///
/// `ResultStrip` binds ←/→ as window-scoped key equivalents, which AppKit
/// checks BEFORE the first responder sees the key. A caret already stood down
/// through `editingText`, but a focused `Slider` or `Stepper` -- Steps,
/// Guidance, Length, Strength, identity Weight, Start step, every LoRA scale
/// -- adjusts itself with the same two keys and had no way to say so
/// (finding 02#15).
///
/// One gate, decided here, asked by every surface that binds a bare arrow.
enum ArrowKeyClaim {
    /// Whether this responder consumes an unmodified arrow itself.
    ///
    /// Real AppKit types rather than class-name matching, so a test builds the
    /// actual controls. An `NSTextField` being edited hands off to its field
    /// editor (an `NSTextView`), so both are named.
    static func claims(_ responder: NSResponder?) -> Bool {
        switch responder {
        case is NSTextView, is NSTextField, is NSSlider, is NSStepper: true
        default: false
        }
    }

    /// The live answer for the key window.
    @MainActor static var isClaimedNow: Bool {
        claims(NSApp.keyWindow?.firstResponder)
    }
}
