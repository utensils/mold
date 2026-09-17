import Foundation
import MoldClient

/// What each machine is made of, what it is doing with it, and what it can
/// see beside it.
///
/// One store, not three: devices, telemetry and peers are all "what this
/// machine is", they are all fetched per host through `HostStore`, and three
/// stores would mean three registrations, three failure verbs and three
/// composition-root lines for one pane.
@MainActor
@Observable
final class MachineStore {
    /// Not `private`: `MachineStore+Telemetry` reads it too, and `private`
    /// does not cross a file boundary even within one type.
    let hosts: HostStore
    private(set) var devices: [MoldHost.ID: [DeviceInfo]] = [:]
    /// Written from here and from `MachineStore+Telemetry`'s
    /// `watchResources`, so `private(set)` does not cross that file boundary.
    var resources: [MoldHost.ID: ResourceSnapshot] = [:]
    /// Per host, because a peer list is what THAT machine can see on ITS
    /// network -- plato and this Mac are on different ones.
    private(set) var peers: [MoldHost.ID: [DiscoveryPeer]] = [:]
    /// Devices with a lifecycle change in flight. Their switch is inert until
    /// the machine answers, because the answer is the state.
    private(set) var changing: Set<String> = []
    /// At most ONE telemetry stream exists at a time -- see
    /// `MachineStore+Telemetry`. Not `private`: that file reads it, and
    /// `private` does not cross a file boundary even within one type.
    var telemetry: (host: MoldHost.ID, task: Task<Void, Never>)?

    init(hosts: HostStore) {
        self.hosts = hosts
        // Registration is for the life of the app, exactly the shape
        // `LibraryStore.swift` already uses -- there is no per-pane teardown
        // to hang a removal off.
        hosts.onEvent { [weak self] host, event in
            guard case .deviceStateChanged = event else { return }
            Task { await self?.refreshDevices(on: host) }
        }
    }

    /// Devices, then a single resource sample. Prunes machines no longer in
    /// `hosts.hosts` -- `HostStore.forget(_:)` knows nothing about this
    /// store, so it is this store's own job.
    func refresh(_ host: MoldHost.ID) async {
        await refreshDevices(on: host)
        guard let client = hosts.backend(for: host) else { return }
        do {
            resources[host] = try await client.resources()
            hosts.succeeded(on: host, doing: "read its memory use")
        } catch let MoldClientError.http(status, _, _) where status == 503 {
            // "Not yet" -- a machine answers 503 here for about a second
            // after it boots. Not a fault, and not worth a banner.
        } catch {
            hosts.report(error, on: host, doing: "read its memory use")
        }
        prune()
    }

    func refreshDevices(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            devices[host] = try await client.devices().devices
            hosts.succeeded(on: host, doing: "list its GPUs")
        } catch {
            // A machine that cannot answer keeps the rows it last showed --
            // blanking them says every device vanished, when what happened is
            // a bad connection.
            hosts.report(error, on: host, doing: "list its GPUs")
        }
    }

    /// NOT optimistic: a 202 means `draining`, and a person who flipped a
    /// switch and saw it settle instantly into "on" would have been told the
    /// opposite of what happened. The row is replaced with what the machine
    /// answers, on both the 200 and the 202.
    func setDevice(_ device: DeviceInfo, enabled: Bool, on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        changing.insert(device.id)
        defer { changing.remove(device.id) }
        do {
            let answered = try await client.setDevice(device.id, enabled: enabled)
            replace(answered, on: host)
            hosts.succeeded(on: host, doing: "change that GPU")
        } catch {
            hosts.report(error, on: host, doing: "change that GPU")
        }
    }

    private func replace(_ device: DeviceInfo, on host: MoldHost.ID) {
        guard let index = devices[host]?.firstIndex(where: { $0.id == device.id }) else { return }
        devices[host]?[index] = device
    }

    func refreshPeers(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            peers[host] = try await client.peers()
            hosts.succeeded(on: host, doing: "look for machines near it")
        } catch {
            hosts.report(error, on: host, doing: "look for machines near it")
        }
    }

    func devices(on host: MoldHost.ID) -> [DeviceInfo] { devices[host] ?? [] }
    func resource(on host: MoldHost.ID) -> ResourceSnapshot? { resources[host] }

    /// The live figures for one card when a snapshot has arrived, matched
    /// within one host by ordinal. Absent before the first frame, which is
    /// what makes the row fall back to the device's own figures.
    func sample(for device: DeviceInfo, on host: MoldHost.ID) -> GpuSample? {
        guard let ordinal = device.ordinal else { return nil }
        return resources[host]?.gpus.first { $0.ordinal == ordinal }
    }

    func isChanging(_ device: DeviceInfo) -> Bool { changing.contains(device.id) }

    private func prune() {
        let live = Set(hosts.hosts.map(\.id))
        guard devices.contains(where: { !live.contains($0.key) })
            || resources.contains(where: { !live.contains($0.key) })
            || peers.contains(where: { !live.contains($0.key) })
        else { return }
        devices = devices.filter { live.contains($0.key) }
        resources = resources.filter { live.contains($0.key) }
        peers = peers.filter { live.contains($0.key) }
    }
}
