import Foundation

/// One decoder for the whole wire contract.
///
/// mold serializes in snake_case, so converting here removes a `CodingKeys`
/// block from every type -- and with it the chance of a typo that silently
/// decodes a field as `nil` forever.
public enum MoldJSON {
    public static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    public static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()

    /// For the app's OWN files, where both ends are this binary.
    ///
    /// Deliberately strategy-free, because the two above are NOT inverses:
    /// `convertToSnakeCase` knows `baseURL` is an acronym and writes
    /// `base_url`, while `convertFromSnakeCase` reads `base_url` back as
    /// `baseUrl`. That is harmless on the wire -- mold sends snake_case and
    /// nothing round-trips through both -- and silent data loss anywhere the
    /// app writes something and reads it again, which is how the machine list
    /// came back empty after every launch. Local types spell their keys out.
    public static let localDecoder = JSONDecoder()

    public static let localEncoder = JSONEncoder()
}

/// A `RawRepresentable` enum that tolerates values this build has never heard
/// of instead of failing the whole decode.
///
/// mold's servers are versioned independently of this app -- a host can and
/// will send a control mode or an output format added after this binary was
/// built. Throwing there would lose the entire model list over one unknown
/// string, so every open enum on the wire degrades to `.unknown` and the
/// caller decides what that means.
public protocol OpenWireEnum: RawRepresentable, Codable, Hashable, Sendable
where RawValue == String {
    static var unknown: Self { get }
}

public extension OpenWireEnum {
    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = Self(rawValue: raw) ?? .unknown
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}
