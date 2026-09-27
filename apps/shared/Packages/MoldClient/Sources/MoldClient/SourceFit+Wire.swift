import Foundation

// `source_fit`'s wire shape, which is studio's and not this package's usual
// one: the object's OWN keys are camelCase (`alignX`, `alignY`,
// `upscalerModel`, `sourceFit.ts:85-121`) because the server treats the whole
// value as opaque provenance and never renames anything inside it. A
// `CodingKeys` block would be snake-cased by `MoldJSON.encoder` and a print
// this app made would come back to studio's `parseSourceFitPolicy` with its
// alignment silently missing -- so the object is built as a DICTIONARY, whose
// keys `keyEncodingStrategy` deliberately leaves alone.
extension SourceFit: Codable {
    /// One node of the opaque object: a string, or a nested policy.
    private enum Node: Codable {
        case text(String)
        case object([String: Node])

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case let .text(value): try container.encode(value)
            case let .object(value): try container.encode(value)
            }
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let value = try? container.decode(String.self) { self = .text(value); return }
            self = .object(try container.decode([String: Node].self))
        }

        var text: String? { if case let .text(value) = self { return value }; return nil }
        var object: [String: Node]? { if case let .object(value) = self { return value }; return nil }
    }

    private var node: [String: Node] {
        switch self {
        case .padRepaint, .padFit, .lanczosResize:
            return ["mode": .text(mode.rawValue)]
        case let .cropFill(alignX, alignY):
            var fields: [String: Node] = ["mode": .text(mode.rawValue)]
            // Absent reads as centred (`alignOffset`, `sourceFit.ts:166-173`),
            // so the default is spelled out rather than omitted -- this value
            // is provenance, and a Reuse should see what was chosen.
            if let alignX { fields["alignX"] = .text(alignX.rawValue) }
            if let alignY { fields["alignY"] = .text(alignY.rawValue) }
            return fields
        case let .upscaleThenFit(upscalerModel, fit):
            return ["mode": .text(mode.rawValue),
                    "upscalerModel": .text(upscalerModel),
                    "fit": .object(fit.node)]
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(node)
    }

    /// Defensive, exactly as `parseSourceFitPolicy` is: anything that is not
    /// wire-shaped THROWS rather than degrading, so a corrupt or
    /// after-this-build value can never poison a live draft. Callers that
    /// read provenance decode with `try?`.
    public init(from decoder: Decoder) throws {
        let fields = try decoder.singleValueContainer().decode([String: Node].self)
        try self.init(fields: fields, decoder: decoder)
    }

    private init(fields: [String: Node], decoder: Decoder) throws {
        func fail() -> DecodingError {
            DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                                              debugDescription: "not a source-fit policy"))
        }
        guard let raw = fields["mode"]?.text, let mode = SourceFitMode(rawValue: raw) else {
            throw fail()
        }
        switch mode {
        case .padRepaint: self = .padRepaint
        case .padFit: self = .padFit
        case .lanczosResize: self = .lanczosResize
        case .cropFill:
            let alignX = try fields["alignX"].map { node -> SourceFitAlignX in
                guard let value = node.text.flatMap(SourceFitAlignX.init) else { throw fail() }
                return value
            }
            let alignY = try fields["alignY"].map { node -> SourceFitAlignY in
                guard let value = node.text.flatMap(SourceFitAlignY.init) else { throw fail() }
                return value
            }
            self = .cropFill(alignX: alignX, alignY: alignY)
        case .upscaleThenFit:
            guard let model = fields["upscalerModel"]?.text,
                  let nested = fields["fit"]?.object else { throw fail() }
            let fit = try SourceFit(fields: nested, decoder: decoder)
            // Never nested inside itself (`sourceFit.ts:111`).
            guard fit.mode != .upscaleThenFit else { throw fail() }
            self = .upscaleThenFit(upscalerModel: model, fit: fit)
        }
    }
}
