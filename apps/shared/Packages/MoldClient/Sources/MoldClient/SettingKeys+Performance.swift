import Foundation

// Settings ▸ Performance (design "The pane map"): the port the server binds
// and the scheduler's three timing knobs. The scheduler bounds are the
// ENGINE's shared constant, `SCHEDULER_TIMING_MAX_MS` (`config.rs:809`),
// pinned by `theThreeSchedulerBoundsAreTheSharedConstant` -- one match arm
// backs all three (`config_keys.rs:685-696`), not three copies of 30000.
// `server_port` is a plain `.number` -- it is NOT restart-flagged by the
// server (`routes_config.rs:63` only ever sets it for `scheduler.*`), so its
// row draws no "Needs a restart" caption even though changing it plainly
// needs one; that is the server's answer to give, not this client's to guess.
public extension SettingKeys {
    static let performanceServer: [SettingKey] = [
        SettingKey(
            key: "server_port", label: "Port",
            // config_keys.rs:651: parse_u16(raw, 1, 65535, key)
            help: "The port this machine's server listens on.",
            editor: .number(min: 1, max: 65_535, step: nil)),
    ]

    static let performanceScheduler: [SettingKey] = [
        SettingKey(
            key: "scheduler.replan_debounce_ms", label: "Replan debounce",
            help: "How long the scheduler waits after a change before replanning, in milliseconds.",
            editor: .number(min: 0, max: 30_000, step: 100)),
        SettingKey(
            key: "scheduler.replan_max_delay_ms", label: "Replan max delay",
            help: "The longest the scheduler will ever delay a replan, in milliseconds.",
            editor: .number(min: 0, max: 30_000, step: 100)),
        SettingKey(
            key: "scheduler.warm_wait_max_ms", label: "Warm-wait max",
            help: "How long a job will wait for a warm model before falling back to a cold one, in milliseconds.",
            editor: .number(min: 0, max: 30_000, step: 100)),
    ]

    static var performance: [SettingKey] { performanceServer + performanceScheduler }
}
