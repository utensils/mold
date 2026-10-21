import Foundation

/// Bounded per-part editor state. Stage identities travel with their inputs on reorder.
public struct SequenceDraft: Codable, Hashable, Sendable {
    public struct Stage: Codable, Hashable, Sendable, Identifiable {
        public var id: UUID
        public var input: ChainStageInput
        public var prompt: String { get { input.prompt } set { input.prompt = newValue } }
        public var frames: Int { get { input.frames } set { input.frames = newValue } }
        public var transition: String { get { input.transition } set { input.transition = newValue } }
        public var keyframes: [KeyframeCondition]? { get { input.keyframes } set { input.keyframes = newValue } }
        public var generationReferences: [GenerationReference]? {
            get { input.generationReferences } set { input.generationReferences = newValue }
        }
        public init(prompt: String, frames: Int, transition: String = "smooth") {
            id = UUID(); input = .init(prompt: prompt, frames: frames, transition: transition)
        }
    }
    public var stages: [Stage]
    public init(prompt: String, frames: Int) { stages = [.init(prompt: prompt, frames: frames)] }
    public mutating func append() {
        stages.append(.init(prompt: "", frames: stages.last?.frames ?? 97))
    }
    public mutating func remove(id: UUID) {
        guard stages.count > 1 else { return }; stages.removeAll { $0.id == id }
    }
    public mutating func move(from: Int, to: Int) {
        guard stages.indices.contains(from), stages.indices.contains(to), from != to else { return }
        let stage = stages.remove(at: from); stages.insert(stage, at: to)
    }
    public func wireStages() -> [ChainStageInput] { stages.map(\.input) }
    public func request(base: GenerateRequest, limits: ChainLimits, repeatReferences: Bool = true, motionTailFrames: Int? = nil) throws -> AuthoredChainRequest {
        try AuthoredChainRequest(base, stages: wireStages(), limits: limits, motionTailFrames: motionTailFrames, repeatReferences: repeatReferences)
    }
}
