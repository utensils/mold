import Foundation
import MoldClient
import SwiftUI

extension LibraryPane {
    func extendVideo(_ entry: LibraryEntry) {
        guard entry.print.isVideo, entry.print.trashedAt == nil,
              let host = hosts.host(entry.hostID), let owner = generate.durationOwner(host),
              let backend = hosts.backend(for: host.id), hosts.capabilities[host.id]?.canExtendLibraryVideo == true else { return }
        let snapshot = generate.draft, authoring = generate.videoDuration
        let token = UUID(); generate.durationSourceToken = token
        generate.durationSourceLoading = true; generate.durationSourceError = nil
        destination = .generate
        Task {
            defer { if generate.durationSourceToken == token { generate.durationSourceLoading = false } }
            let current = { @MainActor in
                generate.durationSourceToken == token && generate.durationIsCurrent(owner)
                    && generate.draft == snapshot && generate.videoDuration == authoring
            }
            do {
                let source = try await VideoLibraryExtensionSource.inspect(filename: entry.print.filename,
                    title: entry.print.displayName, metadata: entry.print.metadata,
                    owner: owner, backend: backend, current: current)
                guard current() else { return }
                reuseStore.clear(); generate.sequence.startFresh()
                generate.durationSource = source
                generate.videoDuration = source.authoring(seconds: "")
                generate.draft = source.prefilledDraft
                generate.hostID = host.id; generate.machineChoice = host.id
                if let name = source.metadata.model {
                    generate.modelName = name
                    if let model = models.model(named: name, on: host.id) {
                        let selected = source.metadata.pipeline.flatMap { pipeline in
                            model.generationProfile?.recipes.first { $0.requestSelector?.pipeline == pipeline }
                        } ?? model.defaultRecipe
                        generate.adopt(model: model, on: host.id, keepingDraft: true, selectedRecipe: selected)
                        if let temporal = selected?.temporal {
                            generate.videoDuration?.draft.seconds = (try? VideoDurationDraft.initialSeconds(
                                frames: temporal.frames.default, fps: temporal.fps.resolvedRate(generate.draft.fps))) ?? ""
                        }
                    } else {
                        generate.submissionFeedback.begin("The original model is not installed on this machine. Choose or install a compatible video model.", phase: .refused)
                    }
                }
            } catch {
                if current() {
                    generate.durationSourceError = UserFacingError.describe(error)
                    generate.submissionFeedback.begin(UserFacingError.describe(error), phase: .refused)
                }
            }
        }
    }
}
