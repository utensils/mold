//! Catalog REST + scan-queue surface. Mirrors the single-writer pattern
//! used by `crate::downloads::DownloadQueue`: one scan at a time per
//! server, status polled by id.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use mold_catalog::scanner::{run_scan, ScanOptions, ScanReport};
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CatalogScanStatus {
    Pending,
    Running,
    Done {
        total_entries: usize,
        per_family: BTreeMap<String, String>,
    },
    Failed {
        message: String,
    },
}

#[async_trait]
pub trait ScanDriver: Send + Sync + 'static {
    async fn run(&self, opts: ScanOptions) -> ScanReport;
}

/// Production driver. Hits the real Hugging Face + Civitai endpoints.
pub struct LiveScanDriver;

#[async_trait]
impl ScanDriver for LiveScanDriver {
    async fn run(&self, opts: ScanOptions) -> ScanReport {
        run_scan("https://huggingface.co", "https://civitai.com", &opts).await
    }
}

#[derive(Clone)]
pub struct CatalogScanQueue {
    inner: Arc<Mutex<Inner>>,
    notify: Arc<tokio::sync::Notify>,
}

struct Inner {
    pending: Option<(String, ScanOptions)>,
    statuses: BTreeMap<String, CatalogScanStatus>,
    active: Option<String>,
}

impl Default for CatalogScanQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl CatalogScanQueue {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                pending: None,
                statuses: BTreeMap::new(),
                active: None,
            })),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub async fn enqueue(&self, opts: ScanOptions) -> Result<String, String> {
        let mut inner = self.inner.lock().await;
        if inner.active.is_some() || inner.pending.is_some() {
            return Err("a catalog refresh is already in progress".into());
        }
        let id = Uuid::new_v4().to_string();
        inner.pending = Some((id.clone(), opts));
        inner
            .statuses
            .insert(id.clone(), CatalogScanStatus::Pending);
        drop(inner);
        self.notify.notify_one();
        Ok(id)
    }

    pub async fn status(&self, id: &str) -> Option<CatalogScanStatus> {
        self.inner.lock().await.statuses.get(id).cloned()
    }

    pub async fn drive(self, driver: Arc<dyn ScanDriver>) {
        loop {
            self.notify.notified().await;
            let job = {
                let mut inner = self.inner.lock().await;
                let job = inner.pending.take();
                if let Some((id, _)) = &job {
                    inner.active = Some(id.clone());
                    inner
                        .statuses
                        .insert(id.clone(), CatalogScanStatus::Running);
                }
                job
            };
            let Some((id, opts)) = job else { continue };
            let report = driver.run(opts).await;
            let mut summary = BTreeMap::new();
            for (fam, outcome) in &report.per_family {
                summary.insert(fam.as_str().to_string(), format!("{:?}", outcome));
            }
            let mut inner = self.inner.lock().await;
            inner.active = None;
            inner.statuses.insert(
                id.clone(),
                CatalogScanStatus::Done {
                    total_entries: report.total_entries,
                    per_family: summary,
                },
            );
        }
    }
}

#[cfg(test)]
#[path = "catalog_api_test.rs"]
mod catalog_api_test;
