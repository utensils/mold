import Foundation
import MoldClient
@testable import Mold

// M7 S1: config rows, profiles, and mobile pairing. Stored witnesses live in
// `FakeBackend.swift` itself -- an extension cannot declare stored
// properties -- this file is the route implementations that read them.
extension FakeBackend {
    // MARK: - Config

    func config() async throws -> ConfigListing {
        try record("config")
        guard let configListing else { throw notPlanted() }
        return configListing
    }

    @discardableResult
    func setConfig(_ key: String, to value: ConfigScalar) async throws -> ConfigEntry {
        try record("setConfig")
        configWrites.append((key, value))
        let entry = ConfigEntry(key: key, value: value, source: "db")
        applyToPlantedListing(entry)
        return entry
    }

    @discardableResult
    func resetConfig(_ key: String) async throws -> ConfigEntry {
        try record("resetConfig")
        configResets.append(key)
        let entry = ConfigEntry(key: key, value: .null, source: "default")
        applyToPlantedListing(entry)
        return entry
    }

    /// Mutates the planted `configListing` in place, the way a live
    /// server's next `GET /api/config` would reflect a write it just
    /// accepted -- so `ConfigStore`'s re-read after every `set`/`reset`
    /// (`config_sync.rs:674-688`) has something real to see rather than the
    /// same stale rows it started with.
    private func applyToPlantedListing(_ entry: ConfigEntry) {
        guard let listing = configListing else { return }
        var entries = listing.entries
        if let index = entries.firstIndex(where: { $0.key == entry.key }) {
            entries[index] = entry
        } else {
            entries.append(entry)
        }
        configListing = ConfigListing(profile: listing.profile, entries: entries)
    }

    func configProfiles() async throws -> ConfigProfiles {
        try record("configProfiles")
        guard let profilesAnswer else { throw notPlanted() }
        return profilesAnswer
    }

    // MARK: - Pairing

    func pairingSession() async throws -> PairingSession {
        try record("pairingSession")
        guard !pairingSessions.isEmpty else { throw notPlanted() }
        return pairingSessions.removeFirst()
    }

    func pairedClients() async throws -> PairedClients {
        try record("pairedClients")
        guard let pairedClientsAnswer else { throw notPlanted() }
        return pairedClientsAnswer
    }

    func revokePairedClient(_ id: String) async throws {
        try record("revokePairedClient")
        revokedClients.append(id)
    }
}
