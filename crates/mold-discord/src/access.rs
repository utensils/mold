use std::collections::HashSet;
use std::sync::Mutex;

/// Configured allowed roles, parsed from `MOLD_DISCORD_ALLOWED_ROLES`.
///
/// Supports both role IDs (numeric) and role names (case-insensitive).
/// When no roles are configured, all users are allowed.
#[derive(Debug, Clone)]
pub struct AllowedRoles {
    role_ids: HashSet<u64>,
    role_names: HashSet<String>,
    /// If true, no role restriction — everyone can generate.
    pub unrestricted: bool,
}

impl AllowedRoles {
    /// Parse from a comma-separated env var value.
    /// Numeric tokens become role IDs; others become lowercased role names.
    /// `None` or empty string results in unrestricted access.
    pub fn parse(env_value: Option<&str>) -> Self {
        let value = match env_value {
            Some(v) if !v.trim().is_empty() => v,
            _ => {
                return Self {
                    role_ids: HashSet::new(),
                    role_names: HashSet::new(),
                    unrestricted: true,
                }
            }
        };

        let mut role_ids = HashSet::new();
        let mut role_names = HashSet::new();

        for token in value.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            if let Ok(id) = token.parse::<u64>() {
                role_ids.insert(id);
            } else {
                role_names.insert(token.to_lowercase());
            }
        }

        let unrestricted = role_ids.is_empty() && role_names.is_empty();
        Self {
            role_ids,
            role_names,
            unrestricted,
        }
    }

    /// Check whether any of the member's roles match the allowed set.
    /// Each role is provided as `(role_id, role_name)`.
    pub fn check(&self, member_roles: &[(u64, String)]) -> bool {
        if self.unrestricted {
            return true;
        }
        member_roles.iter().any(|(id, name)| {
            self.role_ids.contains(id) || self.role_names.contains(&name.to_lowercase())
        })
    }

    /// Format allowed roles for display in error messages.
    pub fn display_names(&self) -> String {
        let mut parts: Vec<String> = self.role_names.iter().map(|n| format!("@{n}")).collect();
        for id in &self.role_ids {
            parts.push(format!("<@&{id}>"));
        }
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Per-user block list. In-memory, clears on bot restart.
#[derive(Debug)]
pub struct BlockList {
    blocked: Mutex<HashSet<u64>>,
}

impl BlockList {
    pub fn new() -> Self {
        Self {
            blocked: Mutex::new(HashSet::new()),
        }
    }

    pub fn is_blocked(&self, user_id: u64) -> bool {
        let set = self.blocked.lock().unwrap_or_else(|e| e.into_inner());
        set.contains(&user_id)
    }

    pub fn block(&self, user_id: u64) {
        let mut set = self.blocked.lock().unwrap_or_else(|e| e.into_inner());
        set.insert(user_id);
    }

    pub fn unblock(&self, user_id: u64) {
        let mut set = self.blocked.lock().unwrap_or_else(|e| e.into_inner());
        set.remove(&user_id);
    }
}

impl Default for BlockList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_is_unrestricted() {
        let roles = AllowedRoles::parse(None);
        assert!(roles.unrestricted);
        assert!(roles.check(&[]));
    }

    #[test]
    fn parse_empty_string_is_unrestricted() {
        let roles = AllowedRoles::parse(Some(""));
        assert!(roles.unrestricted);

        let roles = AllowedRoles::parse(Some("  "));
        assert!(roles.unrestricted);
    }

    #[test]
    fn parse_ids_and_names() {
        let roles = AllowedRoles::parse(Some("123, artist, 456"));
        assert!(!roles.unrestricted);
        assert!(roles.role_ids.contains(&123));
        assert!(roles.role_ids.contains(&456));
        assert!(roles.role_names.contains("artist"));
        assert_eq!(roles.role_ids.len(), 2);
        assert_eq!(roles.role_names.len(), 1);
    }

    #[test]
    fn check_unrestricted_always_passes() {
        let roles = AllowedRoles::parse(None);
        assert!(roles.check(&[]));
        assert!(roles.check(&[(999, "random".to_string())]));
    }

    #[test]
    fn check_by_id() {
        let roles = AllowedRoles::parse(Some("123"));
        assert!(roles.check(&[(123, "whatever".to_string())]));
        assert!(!roles.check(&[(999, "whatever".to_string())]));
    }

    #[test]
    fn check_by_name_case_insensitive() {
        let roles = AllowedRoles::parse(Some("Artist"));
        assert!(roles.check(&[(1, "artist".to_string())]));
        assert!(roles.check(&[(1, "ARTIST".to_string())]));
        assert!(roles.check(&[(1, "Artist".to_string())]));
        assert!(!roles.check(&[(1, "member".to_string())]));
    }

    #[test]
    fn check_fails_when_no_match() {
        let roles = AllowedRoles::parse(Some("123, artist"));
        assert!(!roles.check(&[(999, "member".to_string())]));
        assert!(!roles.check(&[]));
    }

    #[test]
    fn check_mixed_id_and_name() {
        let roles = AllowedRoles::parse(Some("123, artist"));
        // Match by ID
        assert!(roles.check(&[(123, "member".to_string())]));
        // Match by name
        assert!(roles.check(&[(999, "artist".to_string())]));
    }

    #[test]
    fn parse_trailing_commas() {
        let roles = AllowedRoles::parse(Some(",artist,,"));
        assert!(!roles.unrestricted);
        assert!(roles.role_names.contains("artist"));
        assert_eq!(roles.role_names.len(), 1);
    }

    #[test]
    fn block_and_unblock() {
        let list = BlockList::new();
        assert!(!list.is_blocked(1));
        list.block(1);
        assert!(list.is_blocked(1));
        list.unblock(1);
        assert!(!list.is_blocked(1));
    }

    #[test]
    fn block_list_default_empty() {
        let list = BlockList::new();
        assert!(!list.is_blocked(1));
        assert!(!list.is_blocked(u64::MAX));
    }

    #[test]
    fn block_list_tracks_independently() {
        let list = BlockList::new();
        list.block(1);
        assert!(list.is_blocked(1));
        assert!(!list.is_blocked(2));
    }

    #[test]
    fn display_names_shows_roles() {
        let roles = AllowedRoles::parse(Some("123, artist"));
        let display = roles.display_names();
        assert!(display.contains("@artist"));
        assert!(display.contains("<@&123>"));
    }

    #[test]
    fn display_names_unrestricted() {
        let roles = AllowedRoles::parse(None);
        assert_eq!(roles.display_names(), "none");
    }
}
