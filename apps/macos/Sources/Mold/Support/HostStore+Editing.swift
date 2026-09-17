import Foundation
import MoldClient

// Adding, changing and removing machines. Split from the reachability
// half purely for size.
@MainActor
extension HostStore {


    /// Adds a machine and starts checking it.
    ///
    /// The address is normalized here as well as in the editor, because this
    /// is the door every caller comes through -- the seeded `MOLD_NATIVE_HOSTS`
    /// list included -- and two spellings of one box would otherwise become
    /// two rows whose prints never merge.
    @discardableResult
    func add(name: String, url: URL, apiKey: String?) -> MoldHost {
        let address = HostAddress.normalize(url.absoluteString) ?? url
        let host = MoldHost(
            name: resolvedName(name, for: address),
            baseURL: address,
            apiKey: apiKey
        )
        hosts.append(host)
        store(apiKey, for: host, doing: "save its key")
        persist()
        Task { await refresh(host) }
        return host
    }

    func update(_ host: MoldHost) {
        guard let index = hosts.firstIndex(where: { $0.id == host.id }) else { return }
        var updated = host
        updated.baseURL = HostAddress.normalize(host.baseURL.absoluteString) ?? host.baseURL
        updated.name = resolvedName(host.name, for: updated.baseURL)
        hosts[index] = updated
        // The editor shows the stored key pre-filled, so an emptied field is
        // somebody clearing it on purpose. Written HERE and only here: the
        // list write below never touches a key (review 05-H5).
        store(updated.apiKey, for: updated, doing: "save its key")
        persist()
        // The key or address may have changed, so EVERYTHING we knew is stale
        // -- the fleet identity included: a machine whose address changed that
        // kept the old box's identity would match the new box's authority
        // frame, and the "this is a different library" repair would never fire.
        forget(updated.id)
        reachability[updated.id] = .unknown
        Task { await refresh(updated) }
    }

    /// A machine already in the list at this address, if there is one.
    ///
    /// `excluding` is the row being edited: saving a host without touching its
    /// address must not report the host as a duplicate of itself.
    func host(at url: URL, excluding id: MoldHost.ID? = nil) -> MoldHost? {
        hosts.first { $0.id != id && HostAddress.sameOrigin($0.baseURL, url) }
    }

    /// Nothing in the list is allowed to be nameless, because the sidebar,
    /// the host badges and the Generate picker all print this.
    private func resolvedName(_ name: String, for url: URL) -> String {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed }
        let suggested = HostAddress.suggestedName(for: url)
        return suggested.isEmpty ? HostAddress.displayString(for: url) : suggested
    }

    /// Everything this app held about one machine, in one place -- because
    /// three callers each clearing the subset they remembered is how a
    /// watcher, an export list and a failure banner outlived the machine that
    /// produced them.
    ///
    /// `listeners` is not pruned: it is keyed by registration, not by machine.
    func forget(_ id: MoldHost.ID) {
        reachability[id] = nil
        capabilities[id] = nil
        exportOptions[id] = nil
        instanceIDs[id] = nil
        watchers.removeValue(forKey: id)?.cancel()
        failures.removeAll { $0.host == id }
    }

    func remove(_ host: MoldHost) {
        hosts.removeAll { $0.id == host.id }
        forget(host.id)
        // A removed default cannot outlive its machine in the preferences.
        // Not inside `forget(_:)` itself: `update(_:)` also calls that, to
        // invalidate what the OLD address answered, and an edit must not
        // silently un-default the machine being edited.
        if defaultMachine == host.id { defaultMachine = nil }
        reconcileEventStreams()
        do {
            try HostPersistence.forget(host)
        } catch {
            report(error, on: host.id, doing: "remove its key")
        }
        persist()
    }

    /// One machine's key, written because the person supplied or cleared one.
    /// A failure is REPORTED -- the Keychain's swallowed `OSStatus` is how a
    /// key could go missing with nobody told (review 05-H5).
    private func store(_ apiKey: String?, for host: MoldHost, doing verb: String) {
        do {
            try HostPersistence.setAPIKey(apiKey, for: host.id)
        } catch {
            report(error, on: host.id, doing: verb)
        }
    }

    /// Adds the in-process engine to the machine list.
    ///
    /// NOT persisted: it exists only while this launch has it running, and
    /// writing it to preferences would leave a dead loopback host behind.
    func adoptLocalEngine(_ host: MoldHost) {
        if let index = hosts.firstIndex(where: { $0.id == host.id }) {
            hosts[index] = host
        } else {
            hosts.insert(host, at: 0)
        }
        forget(host.id)
        reachability[host.id] = .unknown
        Task { await refresh(host) }
    }

    func dropLocalEngine() {
        hosts.removeAll { $0.id == MoldEngine.localHostID }
        forget(MoldEngine.localHostID)
        reconcileEventStreams()
    }

    private func persist() {
        // The local engine is a property of this launch, not of the machine
        // list, so it never goes to disk.
        HostPersistence.save(hosts.filter { $0.id != MoldEngine.localHostID })
    }
}
