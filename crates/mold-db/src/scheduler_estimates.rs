use anyhow::{Context, Result};
use rusqlite::params;

use crate::MetadataDb;

#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerEstimateRecord {
    pub estimate_key: String,
    pub device_class: String,
    pub model_fingerprint: String,
    pub work_kind: String,
    pub shape_bucket: String,
    pub execution_fingerprint: String,
    pub sample_count: u64,
    pub ewma_total_ms: f64,
    pub ewma_load_ms: Option<f64>,
    pub vram_high_water_bytes: Option<u64>,
    pub host_high_water_bytes: Option<u64>,
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
                "SELECT estimate_key, device_class, model_fingerprint, work_kind, shape_bucket,
                        execution_fingerprint, sample_count, ewma_total_ms, ewma_load_ms,
                        vram_high_water_bytes, host_high_water_bytes, last_observed_at
                 FROM scheduler_estimates
                 ORDER BY estimate_key",
            )?;
            let rows = statement.query_map([], |row| {
                Ok(SchedulerEstimateRecord {
                    estimate_key: row.get(0)?,
                    device_class: row.get(1)?,
                    model_fingerprint: row.get(2)?,
                    work_kind: row.get(3)?,
                    shape_bucket: row.get(4)?,
                    execution_fingerprint: row.get(5)?,
                    sample_count: row.get::<_, i64>(6)?.max(0) as u64,
                    ewma_total_ms: row.get(7)?,
                    ewma_load_ms: row.get(8)?,
                    vram_high_water_bytes: row
                        .get::<_, Option<i64>>(9)?
                        .map(|value| value.max(0) as u64),
                    host_high_water_bytes: row
                        .get::<_, Option<i64>>(10)?
                        .map(|value| value.max(0) as u64),
                    last_observed_at: row.get(11)?,
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
                    estimate_key, device_class, model_fingerprint, work_kind, shape_bucket,
                    execution_fingerprint, sample_count, ewma_total_ms, ewma_load_ms,
                    vram_high_water_bytes, host_high_water_bytes, last_observed_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
                 ON CONFLICT(estimate_key) DO UPDATE SET
                    device_class = excluded.device_class,
                    model_fingerprint = excluded.model_fingerprint,
                    work_kind = excluded.work_kind,
                    shape_bucket = excluded.shape_bucket,
                    execution_fingerprint = excluded.execution_fingerprint,
                    sample_count = excluded.sample_count,
                    ewma_total_ms = excluded.ewma_total_ms,
                    ewma_load_ms = excluded.ewma_load_ms,
                    vram_high_water_bytes = excluded.vram_high_water_bytes,
                    host_high_water_bytes = excluded.host_high_water_bytes,
                    last_observed_at = excluded.last_observed_at",
                params![
                    record.estimate_key,
                    record.device_class,
                    record.model_fingerprint,
                    record.work_kind,
                    record.shape_bucket,
                    record.execution_fingerprint,
                    i64::try_from(record.sample_count).unwrap_or(i64::MAX),
                    record.ewma_total_ms,
                    record.ewma_load_ms,
                    record
                        .vram_high_water_bytes
                        .map(|value| i64::try_from(value).unwrap_or(i64::MAX)),
                    record
                        .host_high_water_bytes
                        .map(|value| i64::try_from(value).unwrap_or(i64::MAX)),
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
