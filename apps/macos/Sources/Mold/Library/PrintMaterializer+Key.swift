import Foundation
import MoldClient

// What identifies a print's BYTES on this disk. Split from the cache for
// size when the mesh poster gave one print two files.
extension PrintMaterializer {
    /// The identity of a print's bytes, as a directory name.
    ///
    /// The filename stays OUT of the key and inside the directory, so what
    /// Quick Look titles and what the Finder receives is the print's own name
    /// rather than a hash. `media_version` is absent on older servers; the
    /// timestamp stands in, which at worst re-downloads once.
    static func key(for entry: LibraryEntry) -> String {
        let version = entry.print.mediaVersion ?? String(entry.print.timestamp)
        // `media_version` is the machine's string too, and this is the only
        // place the app makes a DIRECTORY out of one -- so it is folded rather
        // than refused (a print with an odd version is still a print) by the
        // same rule the filename is judged against. mold's own versions carry
        // a colon, which is legal in a POSIX component and invisible here but
        // which the Finder renders as "/", so it goes either way.
        // Room for the machine's UUID and the dash: the whole thing is ONE
        // component, and one byte over is a directory that cannot be created.
        let safe = SafeFilename.folded(version, fallback: String(entry.print.timestamp),
                                       limit: SafeFilename.maxBytes - 37)
        return "\(entry.hostID.uuidString)-\(safe)"
    }
}
