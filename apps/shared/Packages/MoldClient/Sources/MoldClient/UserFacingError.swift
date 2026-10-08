import Foundation
import OSLog

/// Shared native presentation boundary, also covering replies from older servers.
/// Keep the original error for diagnostics; never classify a retry by this text.
public enum UserFacingError {
    private static let logger = Logger(subsystem: "io.utensils.mold", category: "errors")

    /// OSLog keeps raw local diagnostics in private fields; domain and code
    /// remain visible without exposing private paths, URLs or credentials.
    public static func describe(_ error: any Error) -> String {
        let raw = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        let system = error as NSError
        logger.error("Action failed (\(system.domain, privacy: .public), \(system.code, privacy: .public)): \(raw, privacy: .private)")
        return error is MoldClientError ? message(raw) : localMessage(raw)
    }

    public static func localMessage(_ diagnostic: String) -> String {
        let raw = diagnostic.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = raw.lowercased()
        if lower.contains("no space left on device") {
            return "This device is out of storage. Free some disk space and try again."
        }
        if lower.contains("no such file or directory") {
            return "That file is no longer available. Choose it again."
        }
        if lower.contains("permission denied") {
            return "This device cannot access that file. Check its permissions."
        }
        if raw.isEmpty || raw.contains("\n") || raw.count > 240 || lower.contains("operation couldn’t be completed") || lower.contains("operation couldn't be completed") {
            return "That action failed. Check the app’s logs for details."
        }
        return readableByteTokens(raw)
    }

    public static func bytes(_ value: UInt64) -> String {
        let scale: Double
        let unit: String
        switch value {
        case 1_000_000_000_000...: scale = 1_000_000_000_000; unit = "TB"
        case 1_000_000_000...: scale = 1_000_000_000; unit = "GB"
        case 1_000_000...: scale = 1_000_000; unit = "MB"
        case 1_000...: scale = 1_000; unit = "KB"
        default: return "\(value) B"
        }
        var amount = String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), Double(value) / scale)
        while amount.last == "0" { amount.removeLast() }
        if amount.last == "." { amount.removeLast() }
        return "\(amount) \(unit)"
    }

    public static func message(_ diagnostic: String) -> String {
        let raw = diagnostic.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = raw.lowercased()
        if lower.contains("cuda"), lower.contains("restart") || lower.contains("quarantined") { return "The graphics device needs to recover. Restart Mold on that machine before trying again." }
        if let summary = legacyMemorySummary(lower) { return summary }
        if lower.contains("cuda"), lower.contains("cooldown") { return "The machine is recovering from a memory failure. Wait for the cooldown or try a smaller output size." }
        if lower.contains("requires cuda compute capability") { return "This graphics device cannot run that model. Choose another machine or model." }
        if lower.contains("lora adapter is no longer readable") { return "A LoRA file is missing or unreadable. Restore it on the machine, then retry the job." }
        if lower.contains("ffprobe is required") || lower.contains("ffmpeg is required") { return "Video processing tools are missing on the machine. Install ffmpeg and ffprobe, then try again." }
        if lower.contains("admission sample"), lower.contains("device bytes") || lower.contains("host bytes") || lower.contains("unified-memory bytes"),
           let required = number(after: "needs at least ", in: lower),
           let available = number(after: "exceeding the ", in: lower), required > available {
            let resource = lower.contains("device bytes") ? "graphics" : lower.contains("unified-memory bytes") ? "shared" : "system"
            return "Not enough \(resource) memory. Estimated need: \(bytes(required)); available budget: \(bytes(available)) (\(bytes(required - available)) short). Close apps on that machine or try a smaller model or output size."
        }
        if ["out of memory", "out_of_memory", "insufficient memory", "allocation failed", "memory allocation", "cuda_error_oom"].contains(where: lower.contains) {
            return "The machine ran out of memory. Try a smaller model, output size or batch."
        }
        if lower.contains("no space left on device") {
            return "The machine is out of storage. Free some disk space and try again."
        }
        if lower.contains("no such file or directory") {
            return "A required file is missing on the machine. Restore or download the model again."
        }
        if lower.contains("permission denied") {
            return "The machine cannot access a required file. Check its file permissions."
        }
        if lower.contains("no device could produce an execution plan") {
            return "This machine cannot run those settings. Try another machine, model or output size."
        }
        if ["tensor shape", "tensor mismatch", "tensor(", "cuda", "metal error", "backtrace", "panic", "safetensors error", "dtype", "shape mismatch"].contains(where: lower.contains) {
            return "The render failed. Check the machine’s logs for details."
        }
        if raw.isEmpty || raw.contains("\n") || raw.count > 240 {
            return "The request could not be completed. Check the machine’s logs for details."
        }
        return readableByteTokens(raw)
    }

    private static func scaledPrefix(_ raw: Substring) -> UInt64? {
        let parts = raw.split(whereSeparator: \.isWhitespace)
        guard parts.count >= 2, let amount = Double(parts[0].drop(while: { $0 == "~" })) else { return nil }
        let unit = parts[1].trimmingCharacters(in: .punctuationCharacters)
        let scales: [String: Double] = ["tb": 1e12, "gb": 1e9, "mb": 1e6, "kb": 1e3, "bytes": 1, "byte": 1]
        guard let scale = scales[unit] else { return nil }
        let value = amount * scale
        guard value.isFinite, value >= 0, value < Double(UInt64.max) else { return nil }
        return UInt64(value.rounded())
    }

    private static func scaledAfter(_ marker: String, in raw: String) -> UInt64? {
        guard let range = raw.range(of: marker) else { return nil }
        return scaledPrefix(raw[range.upperBound...])
    }

    private static func legacyMemorySummary(_ raw: String) -> String? {
        guard raw.contains("needs more host memory") || raw.contains("needs more device memory") || raw.contains("effective vram capacity"),
              let required = scaledAfter("requires ", in: raw) ?? scaledAfter("needs ~", in: raw) else { return nil }
        var available = scaledAfter("over this request's ~", in: raw) ?? scaledAfter("over the ", in: raw)
        if available == nil, let range = raw.range(of: "requires "), let comma = raw[range.upperBound...].firstIndex(of: ",") {
            available = scaledPrefix(raw[raw.index(after: comma)...])
        }
        guard let available, required > available else { return nil }
        let resource = raw.contains("host memory") ? "system" : raw.contains("metal:") || raw.contains("unified-memory") ? "shared" : "graphics"
        let advice = raw.contains("cooldown") ? "Wait for the cooldown or try a smaller output size." : "Close apps on that machine or try a smaller model or output size."
        return "Not enough \(resource) memory. Estimated need: \(bytes(required)); available budget: \(bytes(available)) (\(bytes(required - available)) short). \(advice)"
    }

    private static func readableByteTokens(_ raw: String) -> String {
        var result = raw
        let expression = try! NSRegularExpression(pattern: #"(?<![\w.])([0-9]+) bytes?\b"#)
        for match in expression.matches(in: raw, range: NSRange(raw.startIndex..., in: raw)).reversed() {
            guard let digits = Range(match.range(at: 1), in: result),
                  let range = Range(match.range, in: result), let value = UInt64(result[digits]) else { continue }
            result.replaceSubrange(range, with: bytes(value))
        }
        return result
    }

    private static func number(after marker: String, in raw: String) -> UInt64? {
        guard let range = raw.range(of: marker), let token = raw[range.upperBound...].split(whereSeparator: \.isWhitespace).first else { return nil }
        return UInt64(token)
    }

    public static func http(status: Int, code: String?, diagnostic: String?) -> String {
        if let diagnostic, !diagnostic.isEmpty {
            let friendly = message(diagnostic)
            if friendly != diagnostic.trimmingCharacters(in: .whitespacesAndNewlines) { return friendly }
            if status < 500 { return friendly }
        }
        switch status {
        case 401, 403: return "The machine did not accept the API key. Check it under Machines."
        case 404: return "That model, job or file is no longer available on the machine."
        case 408, 504: return "The machine took too long to respond. Try again."
        case 413: return "That input is too large. Try a smaller file or output size."
        case 429: return "The machine is busy. Try again shortly."
        default:
            if code == "QUEUE_FULL" { return "The queue is full. Try again shortly." }
            if code == "SERVER_RESTARTING" { return "The machine is restarting. Try again shortly." }
            return diagnostic.map(message) ?? "The request could not be completed. Check the machine’s logs for details."
        }
    }
}
