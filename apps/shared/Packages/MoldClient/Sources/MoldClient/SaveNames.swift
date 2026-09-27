import Foundation

/// What to call each file when several prints are saved into one folder.
///
/// Two rules, both the Finder's, and both about never destroying something
/// that is already there:
///
/// - A name already in the folder gets ` 2`, ` 3`, … before its extension.
///   The multi-print save used to `removeItem` at the destination and copy
///   over it, with no overwrite prompt -- the `NSSavePanel` arm asks, the
///   folder arm never did, so somebody's own `robot.png` was deleted.
/// - Two prints in one selection can COLLIDE with each other even though
///   their names differ: `Robot.png` and `robot.png` are one file on a
///   case-insensitive volume, which APFS is by default, and `café.png` in NFC
///   and NFD are one file on any of them. Ten prints saved, nine files, no
///   message. So names already taken in this batch are compared the way the
///   volume compares them.
public struct SaveNames: Sendable {
    /// Folded the way a case- and normalization-insensitive volume folds a
    /// name, which is the comparison that decides whether two files are one.
    private var taken: Set<String> = []

    public init(existing: [String] = []) {
        for name in existing { taken.insert(Self.fold(name)) }
    }

    public static func fold(_ name: String) -> String {
        name.precomposedStringWithCanonicalMapping.lowercased()
    }

    /// The name to write this print under, and never one already spoken for.
    public mutating func claim(_ filename: String) -> String {
        let name = filename as NSString
        let stem = name.deletingPathExtension
        let suffix = name.pathExtension.isEmpty ? "" : ".\(name.pathExtension)"
        var candidate = filename
        var index = 1
        while taken.contains(Self.fold(candidate)) {
            index += 1
            candidate = "\(stem) \(index)\(suffix)"
        }
        taken.insert(Self.fold(candidate))
        return candidate
    }
}
