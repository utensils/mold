import Foundation

/// The one sentence a row says about where it has got to.
///
/// Ported from `studio/api/activity.ts:31-47`, and shared for the reason
/// stated there: a job must not read "Encoding" on one surface while the same
/// snapshot is reduced to a generic phase on another.
public extension ActiveWorkItem {
    var phaseLabel: String {
        if phase == "preparing", let component = preparationProgress?.component,
           !component.isEmpty {
            return "Preparing · \(component)"
        }
        let head = stage?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
            ?? phase.replacingOccurrences(of: "_", with: " ")
        guard let current, let total, total > 0 else { return head }
        return "\(head) · \(progressText(current: current, total: total))"
    }

    /// Bytes for the two cases where `current`/`total` ARE bytes -- a
    /// download, and a generation loading its weights -- and a plain count
    /// everywhere else. A chain's counter is stages, not bytes
    /// (`activity.ts:36-43`).
    private func progressText(current: Int, total: Int) -> String {
        guard kind == "download"
            || (kind == "generation" && execution != "chain" && phase == "loading")
        else { return "\(current)/\(total)" }
        // The app's own byte convention (decimal, `.file`), which every other
        // size in this binary already uses -- deliberately the platform
        // formatter rather than a second hand-rolled one.
        return "\(Int64(current).formatted(.byteCount(style: .file)))"
            + " / \(Int64(total).formatted(.byteCount(style: .file)))"
    }

    /// What the row is, in the app's words. An unknown kind says the kind
    /// rather than nothing -- a machine doing something this build has never
    /// heard of is still doing it.
    var kindLabel: String {
        if execution == "chain" { return "Generation" }
        switch kind {
        case "generation": return "Generation"
        case "sequence": return "Sequence"
        case "download": return "Download"
        case "prompt_expansion": return "Prompt rewrite"
        case "standalone_upscale", "post_upscale": return "Upscale"
        case "admin_model_load": return "Loading a style"
        case "admin_model_unload": return "Unloading a style"
        default: return kind.replacingOccurrences(of: "_", with: " ").capitalizedFirst
        }
    }
}

extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }

    var capitalizedFirst: String {
        guard let first else { return self }
        return first.uppercased() + dropFirst()
    }
}
