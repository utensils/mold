import Foundation

/// A model family's identifying colour, as sRGB components in 0...1. The
/// table is `mold_core::model_presentation::FAMILY_COLORS`, generated into
/// `ModelPresentationTable.familyColors`; each app turns it into its own
/// colour type, so this stays Foundation-only.
public struct FamilyColor: Hashable, Sendable {
    public let rgb: UInt32
    public var red: Double { Double((rgb >> 16) & 0xFF) / 255 }
    public var green: Double { Double((rgb >> 8) & 0xFF) / 255 }
    public var blue: Double { Double(rgb & 0xFF) / 255 }
    public init(rgb: UInt32) { self.rgb = rgb }
}
