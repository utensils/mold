- **Restore Nix builds after the API-only toggle.** Initialize `web_ui_enabled`
  in explicit CLI test fixtures and remove the stale static-key count assertion
  so package checks can run with the new configuration field.
