import Foundation

extension QueueEntry {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if container.contains(.modelDownload) {
            modelDownload = (try? container.decode(ModelDownload.self, forKey: .modelDownload)) ?? ModelDownload(model: nil)
        }
        stall = try container.decodeIfPresent(StallDiagnostic.self, forKey: .stall)
        id = try container.decode(String.self, forKey: .id)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        metadata = try container.decodeIfPresent(OutputMetadata.self, forKey: .metadata)
        seedPinned = try container.decodeIfPresent(Bool.self, forKey: .seedPinned)
        modelDisplayName = try container.decodeIfPresent(String.self, forKey: .modelDisplayName)
        state = try container.decode(QueueState.self, forKey: .state)
        position = try container.decodeIfPresent(Int.self, forKey: .position)
        startedAtUnixMs = try container.decodeIfPresent(Int.self, forKey: .startedAtUnixMs)
        heldReason = try container.decodeIfPresent(String.self, forKey: .heldReason)
        error = try container.decodeIfPresent(String.self, forKey: .error)
        errorDetail = try container.decodeIfPresent(String.self, forKey: .errorDetail)
        retryable = try container.decodeIfPresent(Bool.self, forKey: .retryable)
        durable = try container.decodeIfPresent(Bool.self, forKey: .durable)
        batchId = try container.decodeIfPresent(String.self, forKey: .batchId)
        clientBatchId = try container.decodeIfPresent(String.self, forKey: .clientBatchId)
        dispatchAttempts = try container.decodeIfPresent(Int.self, forKey: .dispatchAttempts)
        gpu = try container.decodeIfPresent(Int.self, forKey: .gpu)
        targetGpu = try container.decodeIfPresent(Int.self, forKey: .targetGpu)
        batchIndex = try container.decodeIfPresent(Int.self, forKey: .batchIndex)
        explicitlyPaused = try container.decodeIfPresent(Bool.self, forKey: .explicitlyPaused)
        replayed = try container.decodeIfPresent(Bool.self, forKey: .replayed)
    }
}
