use anyhow::{Context, Result};
use rusqlite::params;

use crate::MetadataDb;

#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerEstimateRecord {
    pub estimate_key: String,
    pub device_class: String,
    pub model_family: String,
    pub model_fingerprint: String,
    pub work_kind: String,
    pub shape_bucket: String,
    pub execution_fingerprint: String,
    pub sample_count: u64,
    pub ewma_total_ms: f64,
    pub ewma_runtime_ms: Option<f64>,
    pub ewma_load_ms: Option<f64>,
    pub ewma_warm_reload_ms: Option<f64>,
    pub ewma_prompt_encode_ms: Option<f64>,
    pub ewma_denoise_ms: Option<f64>,
    pub ewma_vae_ms: Option<f64>,
    pub ewma_visual_decode_ms: Option<f64>,
    pub ewma_audio_decode_ms: Option<f64>,
    pub ewma_mux_ms: Option<f64>,
    pub ewma_upscale_ms: Option<f64>,
    pub vram_high_water_bytes: Option<u64>,
    pub host_high_water_bytes: Option<u64>,
    pub failure_count: u64,
    pub invalidated_count: u64,
    pub last_outcome: String,
    pub last_fallback_reason: Option<String>,
    pub last_invalidated_plan_reason: Option<String>,
    pub last_observed_at: i64,
}

pub struct SchedulerEstimates<'a> {
    db: &'a MetadataDb,
}

impl<'a> SchedulerEstimates<'a> {
    pub fn new(db: &'a MetadataDb) -> Self {
        Self { db }
    }

    pub fn list(&self) -> Result<Vec<SchedulerEstimateRecord>> {
        self.db.with_conn(|conn| {
            let mut statement = conn.prepare(
                "SELECT estimate_key, device_class, model_family, model_fingerprint, work_kind, shape_bucket,
                        execution_fingerprint, sample_count, ewma_total_ms, ewma_runtime_ms, ewma_load_ms,
                        ewma_warm_reload_ms, ewma_prompt_encode_ms, ewma_denoise_ms,
                        ewma_vae_ms, ewma_visual_decode_ms, ewma_audio_decode_ms,
                        ewma_mux_ms, ewma_upscale_ms,
                        vram_high_water_bytes, host_high_water_bytes,
                        failure_count, invalidated_count, last_outcome,
                        last_fallback_reason, last_invalidated_plan_reason, last_observed_at
                 FROM scheduler_estimates
                 ORDER BY estimate_key",
            )?;
            let rows = statement.query_map([], |row| {
                Ok(SchedulerEstimateRecord {
                    estimate_key: row.get(0)?,
                    device_class: row.get(1)?,
                    model_family: row.get(2)?,
                    model_fingerprint: row.get(3)?,
                    work_kind: row.get(4)?,
                    shape_bucket: row.get(5)?,
                    execution_fingerprint: row.get(6)?,
                    sample_count: row.get::<_, i64>(7)?.max(0) as u64,
                    ewma_total_ms: row.get(8)?,
                    ewma_runtime_ms: row.get(9)?,
                    ewma_load_ms: row.get(10)?,
                    ewma_warm_reload_ms: row.get(11)?,
                    ewma_prompt_encode_ms: row.get(12)?,
                    ewma_denoise_ms: row.get(13)?,
                    ewma_vae_ms: row.get(14)?,
                    ewma_visual_decode_ms: row.get(15)?,
                    ewma_audio_decode_ms: row.get(16)?,
                    ewma_mux_ms: row.get(17)?,
                    ewma_upscale_ms: row.get(18)?,
                    vram_high_water_bytes: row
                        .get::<_, Option<i64>>(19)?
                        .map(|value| value.max(0) as u64),
                    host_high_water_bytes: row
                        .get::<_, Option<i64>>(20)?
                        .map(|value| value.max(0) as u64),
                    failure_count: row.get::<_, i64>(21)?.max(0) as u64,
                    invalidated_count: row.get::<_, i64>(22)?.max(0) as u64,
                    last_outcome: row.get(23)?,
                    last_fallback_reason: row.get(24)?,
                    last_invalidated_plan_reason: row.get(25)?,
                    last_observed_at: row.get(26)?,
                })
            })?;
            rows.collect::<Result<Vec<_>, _>>()
                .context("reading scheduler estimates")
        })
    }

    pub fn upsert(&self, record: &SchedulerEstimateRecord) -> Result<()> {
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO scheduler_estimates (
                    estimate_key, device_class, model_family, model_fingerprint, work_kind, shape_bucket,
                    execution_fingerprint, sample_count, ewma_total_ms, ewma_runtime_ms, ewma_load_ms,
                    ewma_warm_reload_ms, ewma_prompt_encode_ms, ewma_denoise_ms,
                    ewma_vae_ms, ewma_visual_decode_ms, ewma_audio_decode_ms,
                    ewma_mux_ms, ewma_upscale_ms,
                    vram_high_water_bytes, host_high_water_bytes,
                    failure_count, invalidated_count, last_outcome,
                    last_fallback_reason, last_invalidated_plan_reason, last_observed_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27)
                 ON CONFLICT(estimate_key) DO UPDATE SET
                    device_class = excluded.device_class,
                    model_family = excluded.model_family,
                    model_fingerprint = excluded.model_fingerprint,
                    work_kind = excluded.work_kind,
                    shape_bucket = excluded.shape_bucket,
                    execution_fingerprint = excluded.execution_fingerprint,
                    sample_count = excluded.sample_count,
                    ewma_total_ms = excluded.ewma_total_ms,
                    ewma_runtime_ms = excluded.ewma_runtime_ms,
                    ewma_load_ms = excluded.ewma_load_ms,
                    ewma_warm_reload_ms = excluded.ewma_warm_reload_ms,
                    ewma_prompt_encode_ms = excluded.ewma_prompt_encode_ms,
                    ewma_denoise_ms = excluded.ewma_denoise_ms,
                    ewma_vae_ms = excluded.ewma_vae_ms,
                    ewma_visual_decode_ms = excluded.ewma_visual_decode_ms,
                    ewma_audio_decode_ms = excluded.ewma_audio_decode_ms,
                    ewma_mux_ms = excluded.ewma_mux_ms,
                    ewma_upscale_ms = excluded.ewma_upscale_ms,
                    vram_high_water_bytes = excluded.vram_high_water_bytes,
                    host_high_water_bytes = excluded.host_high_water_bytes,
                    failure_count = excluded.failure_count,
                    invalidated_count = excluded.invalidated_count,
                    last_outcome = excluded.last_outcome,
                    last_fallback_reason = excluded.last_fallback_reason,
                    last_invalidated_plan_reason = excluded.last_invalidated_plan_reason,
                    last_observed_at = excluded.last_observed_at",
                params![
                    record.estimate_key,
                    record.device_class,
                    record.model_family,
                    record.model_fingerprint,
                    record.work_kind,
                    record.shape_bucket,
                    record.execution_fingerprint,
                    i64::try_from(record.sample_count).unwrap_or(i64::MAX),
                    record.ewma_total_ms,
                    record.ewma_runtime_ms,
                    record.ewma_load_ms,
                    record.ewma_warm_reload_ms,
                    record.ewma_prompt_encode_ms,
                    record.ewma_denoise_ms,
                    record.ewma_vae_ms,
                    record.ewma_visual_decode_ms,
                    record.ewma_audio_decode_ms,
                    record.ewma_mux_ms,
                    record.ewma_upscale_ms,
                    record
                        .vram_high_water_bytes
                        .map(|value| i64::try_from(value).unwrap_or(i64::MAX)),
                    record
                        .host_high_water_bytes
                        .map(|value| i64::try_from(value).unwrap_or(i64::MAX)),
                    i64::try_from(record.failure_count).unwrap_or(i64::MAX),
                    i64::try_from(record.invalidated_count).unwrap_or(i64::MAX),
                    record.last_outcome,
                    record.last_fallback_reason,
                    record.last_invalidated_plan_reason,
                    record.last_observed_at,
                ],
            )?;
            Ok(())
        })
    }

    /// Prune stale rows first, then retain at most `max_buckets` newest rows.
    pub fn prune_before(&self, cutoff_unix_s: i64, max_buckets: usize) -> Result<usize> {
        self.db.with_conn(|conn| {
            let stale = conn.execute(
                "DELETE FROM scheduler_estimates WHERE last_observed_at < ?1",
                [cutoff_unix_s],
            )?;
            let overflow = conn.execute(
                "DELETE FROM scheduler_estimates
                 WHERE estimate_key IN (
                   SELECT estimate_key FROM scheduler_estimates
                   ORDER BY last_observed_at DESC, estimate_key ASC
                   LIMIT -1 OFFSET ?1
                 )",
                [i64::try_from(max_buckets).unwrap_or(i64::MAX)],
            )?;
            Ok(stale + overflow)
        })
    }
}
