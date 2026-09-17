import Foundation

/// Where one accessor's elements sit inside the BIN chunk.
///
/// Port of `accessorLayout` / `readFloats` / `readIndices` in
/// `studio/lib/glb.ts:229-384`. Everything is bounds-checked HERE, before a
/// single byte is read, so the readers below can only ever address bytes the
/// layout has already proved are there.
struct GLBAccessor {
    static let unsignedByte = 5121
    static let unsignedShort = 5123
    static let unsignedInt = 5125
    static let float = 5126

    private static let componentSizes: [Int: Int] = [
        5120: 1, unsignedByte: 1, 5122: 2, unsignedShort: 2, unsignedInt: 4, float: 4,
    ]
    private static let typeComponents: [String: Int] = [
        "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    ]

    let componentType: Int
    let components: Int
    let count: Int
    /// Byte offset of element 0 within the BIN chunk.
    let start: Int
    let stride: Int
    let componentSize: Int

    static func layout(in document: [String: Any], bin: [UInt8], index: Int,
                       label: String) throws -> GLBAccessor {
        let accessor = try GLBDocument.objectAt(
            GLBDocument.array(document, "accessors", "document"), index,
            "\(label) accessor")
        if GLBDocument.present(accessor["sparse"]) != nil {
            throw GLBParseError("the \(label) accessor is sparse, which mold never writes")
        }
        let type = try GLBDocument.string(accessor, "type", "\(label) accessor")
        guard let components = typeComponents[type] else {
            throw GLBParseError("the \(label) accessor has unsupported type \"\(type)\"")
        }
        let componentType = try GLBDocument.int(accessor, "componentType", "\(label) accessor")
        guard let componentSize = componentSizes[componentType] else {
            throw GLBParseError(
                "the \(label) accessor has unsupported componentType \(componentType)")
        }
        let count = try GLBDocument.int(accessor, "count", "\(label) accessor")
        let viewIndex = try GLBDocument.int(accessor, "bufferView", "\(label) accessor")
        let view = try GLBDocument.objectAt(
            GLBDocument.array(document, "bufferViews", "document"), viewIndex,
            "\(label) bufferView")
        let buffer = try GLBDocument.int(view, "buffer", "\(label) bufferView", fallback: 0)
        guard buffer == 0 else {
            throw GLBParseError(
                "the \(label) bufferView points at buffer \(buffer); only the embedded "
                    + "BIN chunk is supported")
        }
        let viewOffset = try GLBDocument.int(view, "byteOffset", "\(label) bufferView",
                                             fallback: 0)
        let viewLength = try GLBDocument.int(view, "byteLength", "\(label) bufferView")
        guard viewOffset + viewLength <= bin.count else {
            throw GLBParseError(
                "the \(label) bufferView ends at byte \(viewOffset + viewLength), past the "
                    + "end of the \(bin.count)-byte BIN chunk")
        }

        let elementSize = components * componentSize
        let stride = try GLBDocument.int(view, "byteStride", "\(label) bufferView",
                                         fallback: elementSize)
        guard stride >= elementSize else {
            throw GLBParseError(
                "the \(label) bufferView has byteStride \(stride), smaller than its "
                    + "\(elementSize)-byte elements")
        }
        let accessorOffset = try GLBDocument.int(accessor, "byteOffset", "\(label) accessor",
                                                 fallback: 0)
        let needed = count == 0 ? 0 : accessorOffset + (count - 1) * stride + elementSize
        guard needed <= viewLength else {
            throw GLBParseError(
                "the \(label) accessor reads \(needed) bytes from a \(viewLength)-byte "
                    + "bufferView, past the end of the buffer")
        }

        return GLBAccessor(componentType: componentType, components: components, count: count,
                           start: viewOffset + accessorOffset, stride: stride,
                           componentSize: componentSize)
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
