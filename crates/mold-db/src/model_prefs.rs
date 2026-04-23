//! Per-model preferences backed by the `model_prefs` table (v4 schema).
//!
//! Each row stores the last-used generation parameters for a specific
//! model. When the user switches between models in the TUI, the outgoing
//! model's current params get snapshotted here; the incoming model's row
//! is loaded back onto `GenerateParams`. That is the "FLUX remembers its
//! settings, SDXL remembers its settings" behavior.
//!
//! All columns are `Option<T>` — a model a user has never touched simply
//! has no row, and loading it returns `None` so the caller can stay on
//! the model's hard-coded defaults.

use anyhow::Result;
use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::db::MetadataDb;
use crate::settings::resolve_active_profile;

/// A snapshot of all the per-model fields we persist. Mirrors the TUI's
/// `GenerateParams` plus the last prompt/negative pair so users who type a
/// prompt under one model see it again when they come back.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelPrefs {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub guidance: Option<f64>,
    pub scheduler: Option<String>,
    pub seed_mode: Option<String>,
    pub batch: Option<u32>,
    pub format: Option<String>,
    pub lora_path: Option<String>,
    pub lora_scale: Option<f64>,
    pub expand: Option<bool>,
    pub offload: Option<bool>,
    pub strength: Option<f64>,
    pub control_scale: Option<f64>,
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub last_prompt: Option<String>,
    pub last_negative: Option<String>,
}

impl ModelPrefs {
    /// Load prefs for `model` under the active profile (resolved from
    /// [`resolve_active_profile`]). Returns `None` when no row exists yet.
    pub fn load(db: &MetadataDb, model: &str) -> Result<Option<Self>> {
        Self::load_in(db, &resolve_active_profile(db), model)
    }

    /// Load prefs for `model` under an explicit profile.
    pub fn load_in(db: &MetadataDb, profile: &str, model: &str) -> Result<Option<Self>> {
        db.with_conn(|conn| {
            let row = conn
                .query_row(
                    "SELECT width, height, steps, guidance, scheduler, seed_mode,
                            batch, format, lora_path, lora_scale, expand, offload,
                            strength, control_scale, frames, fps, last_prompt, last_negative
                     FROM model_prefs WHERE profile = ?1 AND model = ?2",
                    params![profile, model],
                    |r| {
                        Ok(ModelPrefs {
                            width: r.get::<_, Option<i64>>(0)?.map(|v| v as u32),
                            height: r.get::<_, Option<i64>>(1)?.map(|v| v as u32),
                            steps: r.get::<_, Option<i64>>(2)?.map(|v| v as u32),
                            guidance: r.get(3)?,
                            scheduler: r.get(4)?,
                            seed_mode: r.get(5)?,
                            batch: r.get::<_, Option<i64>>(6)?.map(|v| v as u32),
                            format: r.get(7)?,
                            lora_path: r.get(8)?,
                            lora_scale: r.get(9)?,
                            expand: r.get::<_, Option<i64>>(10)?.map(|v| v != 0),
                            offload: r.get::<_, Option<i64>>(11)?.map(|v| v != 0),
                            strength: r.get(12)?,
                            control_scale: r.get(13)?,
                            frames: r.get::<_, Option<i64>>(14)?.map(|v| v as u32),
                            fps: r.get::<_, Option<i64>>(15)?.map(|v| v as u32),
                            last_prompt: r.get(16)?,
                            last_negative: r.get(17)?,
                        })
                    },
                )
                .ok();
            Ok(row)
        })
    }

    /// Persist `self` under `model` in the active profile.
    pub fn save(&self, db: &MetadataDb, model: &str) -> Result<()> {
        self.save_in(db, &resolve_active_profile(db), model)
    }

    /// Persist `self` under `(profile, model)`. Upserts — same
    /// (profile, model) pair overwrites.
    pub fn save_in(&self, db: &MetadataDb, profile: &str, model: &str) -> Result<()> {
        let ts = now_ms();
        db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO model_prefs (
                    profile, model, width, height, steps, guidance, scheduler, seed_mode,
                    batch, format, lora_path, lora_scale, expand, offload,
                    strength, control_scale, frames, fps, last_prompt, last_negative,
                    updated_at_ms
                 ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                    ?16, ?17, ?18, ?19, ?20, ?21
                 )
                 ON CONFLICT(profile, model) DO UPDATE SET
                    width = excluded.width,
                    height = excluded.height,
                    steps = excluded.steps,
                    guidance = excluded.guidance,
                    scheduler = excluded.scheduler,
                    seed_mode = excluded.seed_mode,
                    batch = excluded.batch,
                    format = excluded.format,
                    lora_path = excluded.lora_path,
                    lora_scale = excluded.lora_scale,
                    expand = excluded.expand,
                    offload = excluded.offload,
                    strength = excluded.strength,
                    control_scale = excluded.control_scale,
                    frames = excluded.frames,
                    fps = excluded.fps,
                    last_prompt = excluded.last_prompt,
                    last_negative = excluded.last_negative,
                    updated_at_ms = excluded.updated_at_ms",
                params![
                    profile,
                    model,
                    self.width.map(|v| v as i64),
                    self.height.map(|v| v as i64),
                    self.steps.map(|v| v as i64),
                    self.guidance,
                    self.scheduler,
                    self.seed_mode,
                    self.batch.map(|v| v as i64),
                    self.format,
                    self.lora_path,
                    self.lora_scale,
                    self.expand.map(|v| v as i64),
                    self.offload.map(|v| v as i64),
                    self.strength,
                    self.control_scale,
                    self.frames.map(|v| v as i64),
                    self.fps.map(|v| v as i64),
                    self.last_prompt,
                    self.last_negative,
                    ts,
                ],
            )?;
            Ok(())
        })
    }

    /// List every saved model's prefs under the active profile. Used by
    /// `mold config list` and the migration exporter.
    pub fn list(db: &MetadataDb) -> Result<Vec<(String, ModelPrefs)>> {
        Self::list_in(db, &resolve_active_profile(db))
    }

    /// List every saved model's prefs under an explicit profile.
    pub fn list_in(db: &MetadataDb, profile: &str) -> Result<Vec<(String, ModelPrefs)>> {
        db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT model, width, height, steps, guidance, scheduler, seed_mode,
                        batch, format, lora_path, lora_scale, expand, offload,
                        strength, control_scale, frames, fps, last_prompt, last_negative
                 FROM model_prefs
                 WHERE profile = ?1
                 ORDER BY model",
            )?;
            let rows = stmt.query_map(params![profile], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    ModelPrefs {
                        width: r.get::<_, Option<i64>>(1)?.map(|v| v as u32),
                        height: r.get::<_, Option<i64>>(2)?.map(|v| v as u32),
                        steps: r.get::<_, Option<i64>>(3)?.map(|v| v as u32),
                        guidance: r.get(4)?,
                        scheduler: r.get(5)?,
                        seed_mode: r.get(6)?,
                        batch: r.get::<_, Option<i64>>(7)?.map(|v| v as u32),
                        format: r.get(8)?,
                        lora_path: r.get(9)?,
                        lora_scale: r.get(10)?,
                        expand: r.get::<_, Option<i64>>(11)?.map(|v| v != 0),
                        offload: r.get::<_, Option<i64>>(12)?.map(|v| v != 0),
                        strength: r.get(13)?,
                        control_scale: r.get(14)?,
                        frames: r.get::<_, Option<i64>>(15)?.map(|v| v as u32),
                        fps: r.get::<_, Option<i64>>(16)?.map(|v| v as u32),
                        last_prompt: r.get(17)?,
                        last_negative: r.get(18)?,
                    },
                ))
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        })
    }

    pub fn delete(db: &MetadataDb, model: &str) -> Result<bool> {
        Self::delete_in(db, &resolve_active_profile(db), model)
    }

    pub fn delete_in(db: &MetadataDb, profile: &str, model: &str) -> Result<bool> {
        db.with_conn(|conn| {
            let n = conn.execute(
                "DELETE FROM model_prefs WHERE profile = ?1 AND model = ?2",
                params![profile, model],
            )?;
            Ok(n > 0)
        })
    }

    /// Convenience: delete every row under a profile. Used by `mold config
    /// reset --all` for a specific profile scope.
    pub fn delete_all_in(db: &MetadataDb, profile: &str) -> Result<usize> {
        db.with_conn(|conn| {
            let n = conn.execute(
                "DELETE FROM model_prefs WHERE profile = ?1",
                params![profile],
            )?;
            Ok(n)
        })
    }
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> MetadataDb {
        MetadataDb::open_in_memory().unwrap()
    }

    #[test]
    fn missing_model_returns_none() {
        let db = db();
        assert!(ModelPrefs::load(&db, "nonexistent").unwrap().is_none());
    }

    #[test]
    fn save_then_load_roundtrip() {
        let db = db();
        let prefs = ModelPrefs {
            width: Some(1024),
            height: Some(1024),
            steps: Some(20),
            guidance: Some(3.5),
            scheduler: Some("ddim".into()),
            seed_mode: Some("random".into()),
            batch: Some(2),
            format: Some("png".into()),
            lora_path: Some("/path/lora.safetensors".into()),
            lora_scale: Some(0.75),
            expand: Some(true),
            offload: Some(false),
            strength: Some(0.8),
            control_scale: Some(1.2),
            frames: Some(24),
            fps: Some(30),
            last_prompt: Some("a cat".into()),
            last_negative: Some("blurry".into()),
        };
        prefs.save(&db, "flux-dev:q4").unwrap();
        let loaded = ModelPrefs::load(&db, "flux-dev:q4").unwrap().unwrap();
        assert_eq!(loaded, prefs);
    }

    #[test]
    fn save_is_upsert_keyed_on_model() {
        let db = db();
        let mut p = ModelPrefs {
            width: Some(512),
            ..Default::default()
        };
        p.save(&db, "sd15:fp16").unwrap();
        p.width = Some(768);
        p.save(&db, "sd15:fp16").unwrap();
        let loaded = ModelPrefs::load(&db, "sd15:fp16").unwrap().unwrap();
        assert_eq!(loaded.width, Some(768));
        let rows = ModelPrefs::list(&db).unwrap();
        assert_eq!(rows.len(), 1, "upsert must not create a duplicate row");
    }

    /// The marquee behavioral guarantee: switching models preserves each
    /// model's own settings. This is what today's `TuiSession` blob does
    /// *not* give us.
    #[test]
    fn per_model_settings_survive_model_switch() {
        let db = db();

        // Save FLUX prefs: 1024x1024, 20 steps, guidance 3.5.
        ModelPrefs {
            width: Some(1024),
            height: Some(1024),
            steps: Some(20),
            guidance: Some(3.5),
            ..Default::default()
        }
        .save(&db, "flux-dev:q4")
        .unwrap();

        // Switch to SDXL and save its own prefs: 768x768, 30 steps, g 7.5.
        ModelPrefs {
            width: Some(768),
            height: Some(768),
            steps: Some(30),
            guidance: Some(7.5),
            ..Default::default()
        }
        .save(&db, "sdxl:fp16")
        .unwrap();

        // Switch back to FLUX: must come back at the *FLUX* numbers.
        let flux = ModelPrefs::load(&db, "flux-dev:q4").unwrap().unwrap();
        assert_eq!(flux.width, Some(1024));
        assert_eq!(flux.steps, Some(20));
        assert_eq!(flux.guidance, Some(3.5));

        let sdxl = ModelPrefs::load(&db, "sdxl:fp16").unwrap().unwrap();
        assert_eq!(sdxl.width, Some(768));
        assert_eq!(sdxl.steps, Some(30));
    }

    #[test]
    fn list_returns_sorted_models() {
        let db = db();
        for name in &["zeta", "alpha", "middle"] {
            ModelPrefs {
                width: Some(512),
                ..Default::default()
            }
            .save(&db, name)
            .unwrap();
        }
        let names: Vec<_> = ModelPrefs::list(&db)
            .unwrap()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, vec!["alpha", "middle", "zeta"]);
    }

    /// Item 5: the same model under two profiles stores independent
    /// rows so `mold config --profile portrait set model.flux…` never
    /// clobbers the default profile's row.
    #[test]
    fn save_and_load_isolate_across_profiles() {
        let db = db();
        let default_prefs = ModelPrefs {
            width: Some(1024),
            steps: Some(20),
            ..Default::default()
        };
        let dev_prefs = ModelPrefs {
            width: Some(768),
            steps: Some(40),
            ..Default::default()
        };
        default_prefs
            .save_in(&db, "default", "flux-dev:q4")
            .unwrap();
        dev_prefs.save_in(&db, "dev", "flux-dev:q4").unwrap();

        let d = ModelPrefs::load_in(&db, "default", "flux-dev:q4")
            .unwrap()
            .unwrap();
        let x = ModelPrefs::load_in(&db, "dev", "flux-dev:q4")
            .unwrap()
            .unwrap();
        assert_eq!(d.width, Some(1024));
        assert_eq!(d.steps, Some(20));
        assert_eq!(x.width, Some(768));
        assert_eq!(x.steps, Some(40));
    }

    /// Item 5: `list_in` only returns rows for the requested profile.
    #[test]
    fn list_in_is_scoped_to_profile() {
        let db = db();
        ModelPrefs {
            width: Some(1024),
            ..Default::default()
        }
        .save_in(&db, "default", "flux-dev:q4")
        .unwrap();
        ModelPrefs {
            width: Some(768),
            ..Default::default()
        }
        .save_in(&db, "dev", "sdxl:fp16")
        .unwrap();
        let default_rows = ModelPrefs::list_in(&db, "default").unwrap();
        assert_eq!(default_rows.len(), 1);
        assert_eq!(default_rows[0].0, "flux-dev:q4");
        let dev_rows = ModelPrefs::list_in(&db, "dev").unwrap();
        assert_eq!(dev_rows.len(), 1);
        assert_eq!(dev_rows[0].0, "sdxl:fp16");
    }

    /// Item 5: `delete_all_in` scopes to a single profile.
    #[test]
    fn delete_all_in_only_drops_target_profile_rows() {
        let db = db();
        ModelPrefs {
            width: Some(1024),
            ..Default::default()
        }
        .save_in(&db, "default", "flux-dev:q4")
        .unwrap();
        ModelPrefs {
            width: Some(768),
            ..Default::default()
        }
        .save_in(&db, "dev", "sdxl:fp16")
        .unwrap();
        assert_eq!(ModelPrefs::delete_all_in(&db, "dev").unwrap(), 1);
        assert!(ModelPrefs::load_in(&db, "default", "flux-dev:q4")
            .unwrap()
            .is_some());
        assert!(ModelPrefs::load_in(&db, "dev", "sdxl:fp16")
            .unwrap()
            .is_none());
    }

    #[test]
    fn delete_returns_true_when_removed() {
        let db = db();
        ModelPrefs {
            width: Some(1024),
            ..Default::default()
        }
        .save(&db, "flux-dev:q4")
        .unwrap();
        assert!(ModelPrefs::delete(&db, "flux-dev:q4").unwrap());
        assert!(!ModelPrefs::delete(&db, "flux-dev:q4").unwrap());
        assert!(ModelPrefs::load(&db, "flux-dev:q4").unwrap().is_none());
    }
}
