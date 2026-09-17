import Foundation

// The answers `expand` and `remix` come back with. Split from
// `PromptTransform.swift` (the request side) purely for size.

/// `types.rs:770-776`.
public struct ExpandResponse: Codable, Hashable, Sendable {
    public let original: String
    public let expanded: [String]

    public init(original: String, expanded: [String]) {
        self.original = original
        self.expanded = expanded
    }
}

/// `types.rs:814-818`.
public struct RemixVariant: Codable, Hashable, Sendable {
    public let prompt: String
    public let dimensions: [RemixDimension]

    public init(prompt: String, dimensions: [RemixDimension]) {
        self.prompt = prompt
        self.dimensions = dimensions
    }
}

/// `types.rs:820-828`.
public struct RemixResponse: Codable, Hashable, Sendable {
    public let sourcePrompt: String
    public let rootPrompt: String?
    public let sourceKind: RemixSourceKind
    public let task: ExpandTask
    public let variants: [RemixVariant]

    public init(
        sourcePrompt: String, rootPrompt: String? = nil, sourceKind: RemixSourceKind = .direct,
        task: ExpandTask, variants: [RemixVariant]
    ) {
        self.sourcePrompt = sourcePrompt
        self.rootPrompt = rootPrompt
        self.sourceKind = sourceKind
        self.task = task
        self.variants = variants
    }
}
