import Foundation
import MoldClient
import Testing

@testable import Mold

/// `SettingRow.resolve` (S4a): pure, so a curated pane's per-row rendering
/// decision is askable with no view at all -- the same `DiscoverRow.resolve`
/// idiom `ConfigValueField.Plan` already uses.
struct SettingRowTests {
    private var widthSetting: SettingKey {
        SettingKeys.generationRendering.first { $0.key == "default_width" }!
    }

    private var variantSetting: SettingKey {
        SettingKeys.generationRendering.first { $0.key == "t5_variant" }!
    }

    @Test func aKeyTheMachineNeverListedDrawsNothing() {
        #expect(SettingRow.resolve(widthSetting, entry: nil) == nil)
    }

    @Test func aRefusalRidesTheRow() {
        let entry = FakeFixtures.configEntry("default_width", value: .number(1024), source: "db")
        let plan = SettingRow.resolve(widthSetting, entry: entry, refusal: "Must be between 64 and 8192.")
        #expect(plan?.refusal == "Must be between 64 and 8192.")
    }

    @Test func aChoiceDrawsTheEnginesList() {
        let entry = FakeFixtures.configEntry("t5_variant", value: .string("q8"), source: "db")
        let plan = SettingRow.resolve(variantSetting, entry: entry)
        #expect(plan?.setting.editor == .choice(["auto", "fp16", "q8", "q6", "q5", "q4", "q3"]))
    }

    @Test func aNumberWithAStepDrawsAStepper() {
        let entry = FakeFixtures.configEntry("default_width", value: .number(1024), source: "db")
        let plan = SettingRow.resolve(widthSetting, entry: entry)
        #expect(plan?.showsStepper == true)
    }

    /// **Fails today**: `ConfigValueField.resolve` has always mapped
    /// `isEnvOwned` to a read-only row, but `SettingRow` never consulted it.
    /// 12 of the 24 curated keys carry an env var, so on a machine started
    /// with `MOLD_DEFAULT_WIDTH=768` the Generation pane drew a live stepper;
    /// dragging it PUT, the server answered 403 `ENV_OVERRIDDEN`
    /// (`routes_config.rs:195-201`) and the control snapped back
    /// (review 05-M11).
    @Test func anEnvOwnedRowIsReadOnlyAndSaysWhy() {
        let entry = FakeFixtures.configEntry(
            "default_width", value: .number(768), source: "env", envVar: "MOLD_DEFAULT_WIDTH")
        let plan = SettingRow.resolve(widthSetting, entry: entry)

        #expect(plan?.isEnvOwned == true)
        #expect(plan?.showsStepper == false, "an env-owned row offers no control to drag")
        #expect(SettingRow.envOwnedReason(entry).contains("MOLD_DEFAULT_WIDTH"))
    }

    /// A machine that reports no variable name still says the row is locked.
    @Test func anEnvOwnedRowWithNoNamedVariableStillExplainsItself() {
        let entry = FakeFixtures.configEntry("default_width", value: .number(768), source: "env")
        #expect(SettingRow.resolve(widthSetting, entry: entry)?.isEnvOwned == true)
        #expect(SettingRow.envOwnedReason(entry).contains("environment"))
    }

    @Test func aNumberWithNoStepDrawsNoStepper() {
        // No curated key declares a nil step today -- built directly so the
        // negative case is real rather than assumed.
        let setting = SettingKey(
            key: "default_steps", label: "Steps", help: "",
            editor: .number(min: 1, max: 1000, step: nil))
        let entry = FakeFixtures.configEntry("default_steps", value: .number(20), source: "db")
        let plan = SettingRow.resolve(setting, entry: entry)
        #expect(plan?.showsStepper == false)
    }
}
