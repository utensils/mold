import Foundation

extension VideoRestartController {
    /// Delete only the revision the confirmation described. The source video
    /// in Library is independently owned and remains untouched.
    public func discardSource() async -> Bool {
        guard !isBusy, admission == nil, protocolKind == .chain,
            let source, let sourceJobId, let sourceBackend, let options,
            options.discard?.supported == true
        else { return false }
        let mine = revision
        let capturedCurrent = sourceCurrent
        let current = { @MainActor in mine == self.revision && capturedCurrent() }
        isBusy = true
        error = nil
        defer { isBusy = false }
        do {
            try await verify(source, backend: sourceBackend, current: current)
            guard current() else { throw changed() }
            do {
                try await sourceBackend.discardChainJob(
                    id: sourceJobId,
                    request: .init(sourceInstanceId: source.instanceId, sourceRevision: options.sourceRevision))
            } catch {
                let original = error
                guard current() else { throw changed() }
                // A lost DELETE acknowledgement is reconciled by exact ID,
                // never by replaying a different revision or starting work.
                var confirmedMissing = false
                do {
                    let value = try await sourceBackend.chainJob(id: sourceJobId)
                    guard value.id == sourceJobId else { throw MoldClientError.malformedResponse }
                } catch let probe as MoldClientError {
                    guard case .http(let status, _, _) = probe, status == 404 else { throw original }
                    let verifiedStatus = try await sourceBackend.status()
                    guard current(), verifiedStatus.instanceId == source.instanceId else { throw changed() }
                    confirmedMissing = true
                }
                guard confirmedMissing else { throw original }
            }
            guard current() else { throw changed() }
            notice = "Job recovery data deleted. Saved Library videos remain."
            return true
        } catch {
            if current() { self.error = UserFacingError.describe(error) }
            return false
        }
    }
}
