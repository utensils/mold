//! Tests for the download queue. Run with:
//!   cargo test -p mold-ai-server downloads --lib
//!
//! These tests never touch HuggingFace — they inject a fake `PullDriver` so
//! the queue logic can be exercised in isolation.

use crate::downloads::DownloadQueue;

#[tokio::test]
async fn queue_starts_empty() {
    let queue = DownloadQueue::new_for_test();
    let listing = queue.listing().await;
    assert!(listing.active.is_none());
    assert!(listing.queued.is_empty());
    assert!(listing.history.is_empty());
}

#[tokio::test]
async fn enqueue_unknown_model_errors() {
    let queue = DownloadQueue::new_for_test();
    let err = queue.enqueue("no-such-model:xyz".to_string()).await;
    assert!(err.is_err(), "expected unknown-model error");
}
