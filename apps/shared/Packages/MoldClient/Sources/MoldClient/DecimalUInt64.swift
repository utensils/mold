import Foundation

/// Lossless unsigned protocol counters. Accepts legacy JSON integers, emits
/// decimal strings so arbitrary durations survive clients using JavaScript.
@propertyWrapper public struct DecimalUInt64: Codable, Hashable, Sendable {
    public var wrappedValue: UInt64
    public init(wrappedValue: UInt64) { self.wrappedValue = wrappedValue }
    public init(from decoder: any Decoder) throws {
        let value = try decoder.singleValueContainer()
        if let text = try? value.decode(String.self) {
            guard !text.isEmpty, text.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }), let number = UInt64(text) else {
                throw DecodingError.dataCorruptedError(in: value, debugDescription: "Expected an unsigned decimal integer")
            }
            wrappedValue = number
        } else {
            wrappedValue = try value.decode(UInt64.self)
        }
    }
    public func encode(to encoder: any Encoder) throws {
        var value = encoder.singleValueContainer()
        try value.encode(String(wrappedValue))
    }
}
