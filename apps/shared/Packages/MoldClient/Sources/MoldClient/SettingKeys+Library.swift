import Foundation

// Settings ▸ Library (design "The pane map"): how long a trashed print or a
// held queue row survives, and the gallery's storage-version switch. Bounds
// for the two retention keys are the ENGINE's shared constant,
// `GALLERY_TRASH_RETENTION_MAX_DAYS` (`config.rs:812`), pinned by
// `theTwoRetentionBoundsAreTheSharedConstant` -- both arms cite the same
// symbol rather than each hard-coding 3650.
public extension SettingKeys {
    static let library: [SettingKey] = [
        SettingKey(
            key: "gallery.trash_retention_days", label: "Keep trashed prints for",
            // config_keys.rs:699-702: parse_u32(raw, 0, GALLERY_TRASH_RETENTION_MAX_DAYS, key)
            help: """
                  Days a trashed print stays before the sweeper purges it. \
                  0 keeps them until you empty the trash.
                  """,
            editor: .number(min: 0, max: 3_650, step: 1)),
        SettingKey(
            key: "queue.held_retention_days", label: "Keep held jobs for",
            // config_keys.rs:708-711: parse_u32(raw, 0, GALLERY_TRASH_RETENTION_MAX_DAYS, key)
            help: """
                  Days a held queue row is kept before the sweeper purges it. \
                  0 keeps them until you clear it.
                  """,
            editor: .number(min: 0, max: 3_650, step: 1)),
        SettingKey(
            key: "gallery.authority_log", label: "Delta-log gallery storage",
            // config_keys.rs:31-35's own warning.
            help: """
                  Writes the gallery's append-only version-3 storage log. A mold \
                  older than 0.29 cannot publish to a store that has this on.
                  """,
            editor: .toggle),
    ]
}
