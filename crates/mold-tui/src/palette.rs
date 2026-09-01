//! The ^K command palette — the TUI translation of the GUI surfaces' ⌘K
//! launcher (spec §"Command palette": search + grouped actions, Enter runs
//! the top match, Esc closes).
//!
//! The registry is static and pure: [`all_commands`] lists every command in
//! a stable order (navigation → actions → themes), [`filter_commands`] does
//! the same case-insensitive substring match the prompt-history search
//! uses, and [`command_action`] maps a selection onto an existing
//! [`Action`]. Key handling lives with the other popups in `App` — the
//! palette swallows every key by construction.

use crate::action::{Action, View};
use crate::ui::theme::ThemePreset;

/// Identity of a palette command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandId {
    GoCreate,
    GoLibrary,
    GoModels,
    GoMachines,
    GoSettings,
    ComposeChain,
    ToggleAdvanced,
    ConnectMachine,
    RandomizeSeed,
    ExpandPrompt,
    RetryHeldPrints,
    SearchHistory,
    ReviewLicenses,
    OpenHelp,
    Quit,
    SetTheme(ThemePreset),
}

/// A renderable palette row.
#[derive(Debug, Clone)]
pub struct PaletteCommand {
    pub id: CommandId,
    /// Single-width glyph in the info accent — never emoji.
    pub glyph: &'static str,
    pub label: String,
    /// Right-aligned key hint (empty when the command has no chord).
    pub hint: &'static str,
}

/// Every command in display order: workspaces first (mirroring the tab
/// strip), then actions, then one theme command per preset.
pub fn all_commands() -> Vec<PaletteCommand> {
    let mut cmds = vec![
        cmd(CommandId::GoCreate, "✦", "Create — new print".into(), "1"),
        cmd(CommandId::GoLibrary, "▦", "Go to Library".into(), "2"),
        cmd(CommandId::GoModels, "◈", "Go to Models".into(), "3"),
        cmd(CommandId::GoMachines, "⊟", "Go to Machines".into(), "4"),
        cmd(CommandId::GoSettings, "⚙", "Open Settings".into(), "5"),
        cmd(CommandId::ComposeChain, "▸", "Compose a chain".into(), "c"),
        cmd(
            CommandId::ToggleAdvanced,
            "▸",
            "Toggle Advanced options".into(),
            "A",
        ),
        cmd(
            CommandId::ConnectMachine,
            "⊟",
            "Connect a machine…".into(),
            "",
        ),
        cmd(CommandId::RandomizeSeed, "✦", "Randomize seed".into(), "^R"),
        cmd(CommandId::ExpandPrompt, "✦", "Expand prompt".into(), "^E"),
        cmd(
            CommandId::RetryHeldPrints,
            "↻",
            "Retry held prints".into(),
            "^T",
        ),
        cmd(
            CommandId::SearchHistory,
            "▦",
            "Search prompt history".into(),
            "/",
        ),
        cmd(
            CommandId::ReviewLicenses,
            "§",
            "Review model licenses…".into(),
            "",
        ),
        cmd(CommandId::OpenHelp, "?", "Help — keybindings".into(), "?"),
        cmd(CommandId::Quit, "q", "Quit mold".into(), "q"),
    ];
    for preset in ThemePreset::ALL {
        cmds.push(cmd(
            CommandId::SetTheme(preset),
            "◐",
            format!("Switch to {} theme", preset.label()),
            "",
        ));
    }
    cmds
}

fn cmd(id: CommandId, glyph: &'static str, label: String, hint: &'static str) -> PaletteCommand {
    PaletteCommand {
        id,
        glyph,
        label,
        hint,
    }
}

/// Case-insensitive substring filter over the command labels — the same
/// match model as `history::search`. An empty query returns the full
/// registry in stable order.
pub fn filter_commands(query: &str) -> Vec<PaletteCommand> {
    let q = query.trim().to_lowercase();
    all_commands()
        .into_iter()
        .filter(|c| q.is_empty() || c.label.to_lowercase().contains(&q))
        .collect()
}

/// Map a command onto the [`Action`] it performs.
pub fn command_action(id: CommandId) -> Action {
    match id {
        CommandId::GoCreate => Action::SwitchView(View::Create),
        CommandId::GoLibrary => Action::SwitchView(View::Library),
        CommandId::GoModels => Action::SwitchView(View::Models),
        CommandId::GoMachines => Action::SwitchView(View::Machines),
        CommandId::GoSettings => Action::SwitchView(View::Settings),
        CommandId::ComposeChain => Action::ChainEnter,
        CommandId::ToggleAdvanced => Action::ToggleAdvanced,
        CommandId::ConnectMachine => Action::MachinesConnect,
        CommandId::RandomizeSeed => Action::RandomizeSeed,
        CommandId::ExpandPrompt => Action::ExpandPrompt,
        CommandId::RetryHeldPrints => Action::RetryHeldPrints,
        CommandId::SearchHistory => Action::SearchHistory,
        CommandId::ReviewLicenses => Action::ReviewLicenses,
        CommandId::OpenHelp => Action::ShowHelp,
        CommandId::Quit => Action::Quit,
        CommandId::SetTheme(preset) => Action::SetTheme(preset),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_query_returns_all_in_registry_order() {
        let all = all_commands();
        let filtered = filter_commands("");
        assert_eq!(all.len(), filtered.len());
        assert_eq!(all.len(), 15 + ThemePreset::ALL.len());
        for (a, b) in all.iter().zip(filtered.iter()) {
            assert_eq!(a.id, b.id);
        }
    }

    #[test]
    fn filter_is_case_insensitive_substring() {
        let hits = filter_commands("lib");
        assert!(hits.iter().any(|c| c.id == CommandId::GoLibrary));
        let hits = filter_commands("THEME");
        assert_eq!(hits.len(), ThemePreset::ALL.len());
        let hits = filter_commands("no-such-command-zzz");
        assert!(hits.is_empty());
    }

    #[test]
    fn every_command_maps_to_a_non_none_action() {
        for c in all_commands() {
            assert_ne!(
                command_action(c.id),
                Action::None,
                "{:?} maps to Action::None",
                c.id
            );
        }
    }

    #[test]
    fn theme_commands_cover_all_presets() {
        let theme_cmds: Vec<_> = all_commands()
            .into_iter()
            .filter(|c| matches!(c.id, CommandId::SetTheme(_)))
            .collect();
        assert_eq!(theme_cmds.len(), ThemePreset::ALL.len());
    }

    #[test]
    fn nav_hints_match_bound_keys() {
        for (id, hint) in [
            (CommandId::GoCreate, "1"),
            (CommandId::GoLibrary, "2"),
            (CommandId::GoModels, "3"),
            (CommandId::GoMachines, "4"),
            (CommandId::GoSettings, "5"),
            (CommandId::ComposeChain, "c"),
            (CommandId::ToggleAdvanced, "A"),
        ] {
            let all = all_commands();
            let c = all.iter().find(|c| c.id == id).unwrap();
            assert_eq!(c.hint, hint, "{id:?}");
        }
    }

    #[test]
    fn glyphs_are_single_width_non_emoji() {
        for c in all_commands() {
            assert_eq!(
                unicode_width::UnicodeWidthStr::width(c.glyph),
                1,
                "glyph {} must be single-width",
                c.glyph
            );
        }
    }
}
