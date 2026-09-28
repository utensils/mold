import MoldClient
import MoldMesh

extension MeshViewFailure {
    /// A parse refusal keeps the reader's own sentence, which names what was
    /// wrong with the file rather than saying only that something was. App-side
    /// because the transport wording (`failureSentence`) is the app's.
    @MainActor static func reading(_ error: any Error) -> MeshViewFailure {
        if let parse = error as? GLBParseError { return .unreadable(parse.description) }
        return .transport(error.failureSentence)
    }
}
