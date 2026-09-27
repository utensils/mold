import Foundation

// What was asked for, what answered, and how it was steered.
extension PrintDetails {

    /// A sequence's recorded prompt is every stage newline-joined, so it is
    /// shown as it was recorded -- this is provenance, and joining or trimming
    /// it would describe a render nobody made. `RenderDraft(reusing:)` is the
    /// one place that reduces it to the first stage, because reuse is
    /// authoring a NEW print rather than describing this one.
    static func promptGroup(_ meta: OutputMetadata) -> PrintDetailGroup? {
        group("Prompt", [
            row("Prompt", meta.prompt, isProse: true),
            row("Negative prompt", meta.negativePrompt, isProse: true),
            // Only when a wand actually changed it: otherwise it is the same
            // paragraph printed twice.
            meta.originalPrompt == meta.prompt ? nil
                : row("Before expanding", meta.originalPrompt, isProse: true),
        ])
    }

    static func modelGroup(_ entry: LibraryEntry) -> PrintDetailGroup? {
        let meta = entry.print.metadata
        return group("Model", [
            row("Model", meta.model),
            row("Family", meta.family),
            row("Machine", entry.hostName),
        ])
    }

    static func settingsGroup(_ meta: OutputMetadata) -> PrintDetailGroup? {
        group("Settings", [
            // Never grouped: a seed with separators in it is a seed nobody can
            // paste back.
            row("Seed", meta.seed.map(String.init)),
            row("Steps", meta.steps.map(String.init)),
            row("Guidance", decimal(meta.guidance)),
            row("Scheduler", meta.scheduler.map(schedulerName)),
            row("Strength", decimal(meta.strength, places: 2)),
            row("CFG+", meta.cfgPlus.map { $0 ? "On" : "Off" }),
            row("Flow shift", decimal(meta.sampleShift)),
            row("Distill (high)", decimal(meta.distillStrengthHigh, places: 2)),
            row("Distill (low)", decimal(meta.distillStrengthLow, places: 2)),
        ])
    }

    /// The host sends kebab-case (`types.rs:137`); this is the same name with
    /// the hyphens opened out, so a solver added after this build still reads
    /// as words rather than being dropped for not being on a list.
    static func schedulerName(_ wire: String) -> String {
        wire.replacingOccurrences(of: "-", with: " ")
    }

    static func decimal(_ value: Double?, places: Int = 1) -> String? {
        value.map { $0.formatted(.number.precision(.fractionLength(0 ... places))) }
    }
}
