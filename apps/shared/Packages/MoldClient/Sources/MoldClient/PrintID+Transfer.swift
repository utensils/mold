import CoreTransferable
import UniformTypeIdentifiers

public extension UTType {
    /// A print being dragged WITHIN Mold.
    ///
    /// Its own type rather than plain JSON, so a drop target accepts prints
    /// and not every JSON file on the machine. Declared in the app's
    /// Info.plist under `UTExportedTypeDeclarations`.
    static let moldPrint = UTType(exportedAs: "io.utensils.mold.print")
}

/// A print's identity travels as itself between two places in this app.
///
/// The FILE representation that a drag to the Finder needs lives on
/// `DraggablePrint`, because it has to fetch bytes from another machine.
/// This is the cheap one: an id, for a drop onto a collection.
extension PrintID: Transferable {
    public static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .moldPrint)
    }
}
