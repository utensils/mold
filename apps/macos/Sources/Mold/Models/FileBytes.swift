import Foundation

/// A size ON DISK, in the words the Models pane uses everywhere.
///
/// Three panes each carried a private copy of this one line, and two of them
/// -- a row's own progress and the same download in the Downloads popover --
/// render the identical "3.2 GB of 11.9 GB" sentence for the identical job.
///
/// `.file` on purpose, and NOT `DeviceWords.bytes`, which is `.memory`: a
/// download is a file, and a card's VRAM is memory, so the two round
/// differently by design (see `DeviceWords` and `GeneralSettings`).
enum FileBytes {
    static func text(_ count: Int64) -> String {
        count.formatted(.byteCount(style: .file))
    }

    /// The wire reports sizes unsigned; a value past `Int64.max` is not a
    /// disk, so clamping is the honest read.
    static func text(_ count: UInt64) -> String {
        text(Int64(clamping: count))
    }

    /// What one download has fetched, of what it will.
    static func progress(done: Int64, total: Int64) -> String {
        "\(text(done)) of \(text(total))"
    }
}
