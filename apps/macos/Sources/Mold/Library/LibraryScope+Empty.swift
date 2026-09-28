import MoldClient

extension LibraryScope {
    /// What an empty shelf says on the Mac, where filing is a drag. The phone
    /// words its own (`LibraryScope` itself lives in MoldClient).
    var emptyMessage: String {
        switch self {
        case .all: "Prints from every machine appear here."
        case .favorites: "Stars you add show up here."
        case .collection: "Drag prints onto this collection to file them here."
        case .trash: "Deleted prints wait here until their machine purges them."
        }
    }
}
