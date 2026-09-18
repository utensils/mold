import AppKit
import MoldClient
import SwiftUI

// The doors, and the one import behind them. Split from the well's own shape
// purely for size.
//
// Every door goes through `start`, which CANCELS the one before it: an
// untracked task per pick meant a 48 MP HEIC chosen first could finish last
// and overwrite the small PNG chosen after it. A newer pick always wins.
extension PictureWell {
    /// The rows that open a door are the chooser's; everything else is the
    /// caller's. A well whose menu carries a row no arm here names simply
    /// hands it on, so a new row is a caller's job and never silently dead.
    func route(_ action: GenerateAction) {
        switch action {
        case .chooseFile, .replacePicture, .replacePhoto:
            choose()
        case .chooseFromLibrary, .replaceFromLibrary:
            showsLibrary = true
        case .paste:
            start { intake.paste() }
        default:
            // A destructive row retracts the well's complaint about a file
            // that did not land, and cancels an import still in flight: the
            // picture both were about is being taken away, and a slow one
            // finishing afterwards would refill a well somebody just emptied.
            if action.isDestructive {
                importTask?.cancel()
                importTask = nil
                importFailure = nil
            }
            perform(action)
        }
    }

    /// The Library sheet hands a picture straight to the well, already
    /// conformed to what this well accepts and already base64'd off the main
    /// actor -- so it needs no import task of its own.
    func deliver(_ picked: ImportedPicture) {
        pick(picked)
        importFailure = nil
    }

    func handle(_ drops: [PictureDrop]) {
        start { intake.drops(drops) }
    }

    private func choose() {
        let urls = PictureSource.choose(allowsMultiple: allowsMultiple)
        guard !urls.isEmpty else { return }
        start { intake.files(urls) }
    }

    private func start(_ work: () -> Task<Void, Never>) {
        importTask?.cancel()
        importTask = work()
    }

    /// The one pipeline, wired to this well's own acceptance policy and its
    /// own failure sentence.
    private var intake: PictureIntake {
        PictureIntake(
            accepting: accepting, hosts: hosts, library: library,
            deliver: pick, report: { importFailure = $0 })
    }
}
