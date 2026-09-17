import Foundation

// Reading an accessor's elements out of the BIN chunk, once `layout` has
// proved every byte is there. Split for size; the bounds argument lives
// with the layout, and these only ever address what it admitted.
extension GLBAccessor {
    /// `offset + (count - 1) * stride + elementSize`, saturating at
    /// `Int.max` instead of trapping. Every operand is a file-supplied number
    /// this reader has not yet bounded.
    static func span(count: Int, stride: Int, offset: Int,
                             elementSize: Int) -> Int {
        let (rows, rowsOverflow) = (count - 1).multipliedReportingOverflow(by: stride)
        if rowsOverflow { return .max }
        let (withOffset, offsetOverflow) = rows.addingReportingOverflow(offset)
        if offsetOverflow { return .max }
        let (total, totalOverflow) = withOffset.addingReportingOverflow(elementSize)
        return totalOverflow ? .max : total
    }

    func floats(in bin: [UInt8], label: String) throws -> [Float] {
        guard componentType == Self.float else {
            throw GLBParseError(
                "the \(label) accessor is componentType \(componentType); mold writes "
                    + "float \(label)")
        }
        var out = [Float](repeating: 0, count: count * components)
        for element in 0..<count {
            let row = start + element * stride
            for component in 0..<components {
                out[element * components + component] =
                    Float(bitPattern: try readUInt32(bin, at: row + component * 4))
            }
        }
        return out
    }

    func indices(in bin: [UInt8]) throws -> [UInt32] {
        guard components == 1 else {
            throw GLBParseError("the index accessor must be SCALAR")
        }
        var out = [UInt32](repeating: 0, count: count)
        for element in 0..<count {
            let at = start + element * stride
            switch componentType {
            case Self.unsignedByte:
                guard at < bin.count else {
                    throw GLBParseError("the index accessor reads past the BIN chunk")
                }
                out[element] = UInt32(bin[at])
            case Self.unsignedShort:
                guard at >= 0, at + 2 <= bin.count else {
                    throw GLBParseError("the index accessor reads past the BIN chunk")
                }
                out[element] = UInt32(bin[at]) | UInt32(bin[at + 1]) << 8
            case Self.unsignedInt:
                out[element] = try readUInt32(bin, at: at)
            default:
                throw GLBParseError(
                    "the index accessor has componentType \(componentType); expected an "
                        + "unsigned byte, short or int")
            }
        }
        return out
    }
}
