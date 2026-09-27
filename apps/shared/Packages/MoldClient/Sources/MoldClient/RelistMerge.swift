import Foundation

/// What a freshly-read listing should replace on screen.
///
/// A full listing is a round trip, and a `gallery_added` arriving inside it
/// names a print the answer was composed before -- so assigning the answer
/// wholesale dropped a print that had just landed, with nothing to re-list
/// again afterwards. Reading the rows as they were when the question was ASKED
/// is what tells the two apart:
///
/// - named by the answer: the machine's word, and it wins;
/// - not named, and not there when we asked: it arrived while the answer was
///   in flight, and it stays;
/// - not named, but there when we asked: the machine says it is gone.
public enum RelistMerge {
    public static func merged(answer: [LibraryEntry], onScreen: [LibraryEntry],
                              asked: Set<String>) -> [LibraryEntry] {
        let answered = Set(answer.map(\.print.filename))
        return answer + onScreen.filter {
            !answered.contains($0.print.filename) && !asked.contains($0.print.filename)
        }
    }
}
