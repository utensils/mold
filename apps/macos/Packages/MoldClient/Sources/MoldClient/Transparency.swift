import Foundation

/// `capabilities.transparency` -- the transparent-background contract
/// (`GenerateRequest.transparent_background`).
///
/// `mold_core::generation_profile::transparency_for_recipe` answers it once
/// for the server, admission, the CLI and every GUI; this is the Swift
/// reading, a port of `studio/lib/transparency.ts`. There is no family sniff
/// anywhere on purpose: a toggle an older host would silently DROP would
/// render an opaque picture somebody asked to be transparent.
///
/// The prompt recipe ("This is an RGBA image with transparency. ...") is
/// applied by the ENGINE, never here: the stored prompt, Reuse and the
/// expander all keep the person's own words.
public struct TransparencyCapability: Codable, Hashable, Sendable {
    public let mode: ControlMode
    /// The toggle's default position (always `false` today).
    public let `default`: Bool
    /// Output containers that carry alpha, narrowed to what the binary
    /// encodes. Never JPEG.
    public let formats: [String]
    /// An alpha-carrying REFERENCE keeps alpha in the output even with the
    /// toggle off (Qwen Image 2.1's four-channel VAE).
    public let nativeAlpha: Bool
    /// The server's own sentence for a hidden block.
    public let reason: String?

    /// The toggle to draw, or nil where none may be offered: a `hidden`
    /// recipe, or an adjustable one whose alpha formats the binary cannot
    /// encode at all (`transparencyControl`, `transparency.ts`).
    public var control: TransparencyControl? {
        guard mode == .adjustable, !formats.isEmpty else { return nil }
        return TransparencyControl(formats: formats, nativeAlpha: nativeAlpha)
    }
}

/// What a surface needs to render -- and serialize -- the toggle.
public struct TransparencyControl: Hashable, Sendable {
    public let formats: [String]
    public let nativeAlpha: Bool

    public init(formats: [String], nativeAlpha: Bool) {
        self.formats = formats
        self.nativeAlpha = nativeAlpha
    }

    /// The toggle's label on every surface.
    public static let label = "Transparent background"
    /// The one-line explanation under the toggle.
    public static let note = "Cut the subject out onto a transparent background (PNG or WebP)."
    /// Why JPEG is disabled while the toggle is on: admission refuses the
    /// pair (`validate_transparency_against`) rather than flattening it.
    public static let unavailableFormatReason =
        "JPEG has no transparency, so a transparent background saves as PNG or WebP."
}

public extension RecipeCapabilities {
    /// The toggle this recipe offers, or nil -- an older host included.
    var transparencyControl: TransparencyControl? { transparency?.control }
}
