import MoldClient

/// Where a prompt rewrite stands.
///
/// `.advised` is not a failure and not an offer: a family with no text encoder
/// is answered by the SERVER, at 200, with its own guide's words and without
/// an LLM being created, activated or pulled. Showing that as an error would
/// be this app inventing a refusal the machine did not make.
enum Expansion: Equatable {
    case idle
    case working(Kind)
    case offering(Offer)
    case advised(String)
    case refused(String)
    /// This machine would expand locally but has not got the model. The name
    /// comes from `capabilities.expand.model`, NEVER from the 422's prose --
    /// that message hard-codes `qwen3-expand` whatever the host configured
    /// (`crates/mold-server/src/routes.rs:4010-4019`).
    case needsModel(String)

    enum Kind: Equatable { case expand, remix }

    /// What came back, and what it replaced.
    struct Offer: Equatable {
        let kind: Kind
        let original: String
        /// The task this rewrite is provenance for.
        ///
        /// For a remix this is the server's own `RemixResponse.task`. An
        /// `ExpandResponse` carries no task at all -- the server resolves one
        /// internally (`ExpandTask::for_family`) but never echoes it back --
        /// so an expand offer records the type's own backward-compatible
        /// default (`ExpandTask.textToImage`) rather than this app
        /// replicating the family-name table that resolves it server-side.
        /// `prompt_transform.task` is provenance only (never read back for
        /// behaviour, `types.rs:1993-2000`), so recording the honest default
        /// here is a documented approximation, not a silent wrong answer.
        let task: ExpandTask
        let choices: [Choice]
    }

    /// One rewrite, with the dimensions a remix varied (empty for an expand).
    struct Choice: Equatable, Identifiable {
        let prompt: String
        let dimensions: [RemixDimension]
        var id: String { prompt }
    }
}

/// What the wand shows, decided once so a view never has to ask three
/// questions itself.
enum ExpansionOffer: Equatable {
    /// No model chosen yet, or this host has said outright that it does not
    /// expand prompts.
    case hidden
    case wand(canRemix: Bool)
    /// This host would expand locally but has not pulled the model.
    case needsModel(String)
}

/// What `accept(_:)` puts back on `revertExpansion()`, and what it takes the
/// draft's prompt off of to answer `canRevertExpansion`.
struct LastAcceptedPrompt: Equatable {
    /// What `accept(_:)` wrote to `draft.prompt`. Reverting is only offered
    /// while the draft's prompt is still exactly this -- editing it by hand
    /// is a different intent from "undo the wand".
    let prompt: String
    let previousPrompt: String
    let previousOriginalPrompt: String?
    let previousTransform: PromptTransformProvenance?
}
