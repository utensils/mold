import Foundation

/// One row of `GET /api/config`. `types.rs:12833-12852`.
public struct ConfigEntry: Codable, Hashable, Sendable {
    public let key: String
    /// Scalar or null; null means the key exists in the registry and is unset.
    public let value: ConfigScalar
    /// `"db"`, `"file"`, `"env"` or `"default"`.
    public let source: String
    public let envVar: String?
    public let restartRequired: Bool?

    public init(key: String, value: ConfigScalar, source: String,
                envVar: String? = nil, restartRequired: Bool? = nil) {
        self.key = key
        self.value = value
        self.source = source
        self.envVar = envVar
        self.restartRequired = restartRequired
    }
}

public struct ConfigListing: Codable, Hashable, Sendable {
    public let profile: String?
    public let entries: [ConfigEntry]

    public init(profile: String? = nil, entries: [ConfigEntry]) {
        self.profile = profile
        self.entries = entries
    }
}

/// A config value, which the wire types as string | number | bool | null.
public enum ConfigScalar: Hashable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null

    public var int: Int? {
        guard case let .number(value) = self else { return nil }
        return Int(value)
    }

    public var double: Double? {
        guard case let .number(value) = self else { return nil }
        return value
    }

    public var text: String? {
        guard case let .string(value) = self else { return nil }
        return value
    }
}

extension ConfigScalar: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let bool = try? container.decode(Bool.self) {
            self = .bool(bool)
        } else if let number = try? container.decode(Double.self) {
            self = .number(number)
        } else if let string = try? container.decode(String.self) {
            self = .string(string)
        } else {
            throw DecodingError.typeMismatch(
                ConfigScalar.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "expected string, number, bool or null"))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(value): try container.encode(value)
        case let .number(value): try container.encode(value)
        case let .bool(value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

