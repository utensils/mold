import Foundation

/// `GET /api/config/profiles`. `types.rs:12864-12870`.
public struct ConfigProfiles: Codable, Hashable, Sendable {
    /// `MOLD_PROFILE` wins over the stored row, so this is not necessarily
    /// what anybody last asked for.
    public let active: String
    public let profiles: [String]

    public init(active: String, profiles: [String]) {
        self.active = active
        self.profiles = profiles
    }
}

// No setter type here on purpose: switching a profile writes the
// `profile.active` meta-row without touching the running server's loaded
// config (`routes_config.rs:380-392`), so a table that could switch would
// show one profile's values while an edit landed in another's. Profiles are
// read-only in this app (design decision 9) -- switch with `mold config` and
// restart.
