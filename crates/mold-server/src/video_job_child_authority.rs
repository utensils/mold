//! Public controls act on parent videos, never trusted internal work IDs.
use crate::{routes::ApiError, state::AppState};
use axum::http::StatusCode;

pub(crate) fn owned() -> ApiError {
    ApiError::with_code(
        "This clip belongs to a video. Use the parent video’s Queue controls.",
        "VIDEO_JOB_CHILD_OWNED",
        StatusCode::CONFLICT,
    )
}

pub(crate) fn require_public_chain(db: &mold_db::MetadataDb, id: &str) -> Result<(), ApiError> {
    if mold_db::video_job_child_owners::parent_id(db, id)
        .map_err(|e| ApiError::internal(format!("could not read video ownership: {e:#}")))?
        .is_some()
    {
        return Err(owned());
    }
    Ok(())
}

pub(crate) async fn require_public_generation(state: &AppState, id: &str) -> Result<(), ApiError> {
    let db = state.metadata_db.clone();
    let id = id.to_string();
    tokio::task::spawn_blocking(move || {
        if let Some(db) = db.as_ref().as_ref() {
            if mold_db::chain_stage_generations::owner_for_generation(db, &id)
                .map_err(|e| ApiError::internal(format!("could not read stage ownership: {e:#}")))?
                .is_some()
            {
                return Err(owned());
            }
        }
        Ok(())
    })
    .await
    .map_err(|e| ApiError::internal(format!("ownership check stopped: {e}")))?
}

pub(crate) async fn require_public_batch(state: &AppState, id: &str) -> Result<(), ApiError> {
    let detail = state
        .queue_journal
        .durable_generation_batch(id)
        .map_err(ApiError::internal)?;
    if let Some(detail) = detail {
        for child in detail.children {
            require_public_generation(state, &child.job_id).await?;
        }
    }
    Ok(())
}

pub(crate) async fn suppress_owned_generations(
    state: &AppState,
    entries: &mut Vec<crate::job_registry::JobEntry>,
) -> Result<(), ApiError> {
    let db = state.metadata_db.clone();
    let ids = entries
        .iter()
        .map(|entry| entry.id.clone())
        .collect::<Vec<_>>();
    let hidden = tokio::task::spawn_blocking(
        move || -> anyhow::Result<std::collections::HashSet<String>> {
            let mut hidden = std::collections::HashSet::new();
            if let Some(db) = db.as_ref().as_ref() {
                for id in ids {
                    if mold_db::chain_stage_generations::owner_for_generation(db, &id)?.is_some() {
                        hidden.insert(id);
                    }
                }
            }
            Ok(hidden)
        },
    )
    .await
    .map_err(|e| ApiError::internal(format!("ownership projection stopped: {e}")))?
    .map_err(|e| ApiError::internal(format!("ownership projection failed: {e:#}")))?;
    entries.retain(|entry| !hidden.contains(&entry.id));
    Ok(())
}

/// Caller owns the settled parent's mutation lock. Durable parent state gates
/// new dispatch, so no internal stage can acquire a GPU during this cleanup.
#[cfg(feature = "mp4")]
pub(crate) async fn cleanup_owned_stage_generations(
    state: &AppState,
    parent_id: &str,
) -> Result<(), ApiError> {
    let mut cursor = String::new();
    loop {
        let db = state.metadata_db.clone();
        let parent = parent_id.to_string();
        let after = cursor.clone();
        let page=tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<String>> {
            let db=db.as_ref().as_ref().ok_or_else(||anyhow::anyhow!("video database unavailable"))?;
            db.with_conn(|conn| {
                let running:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM video_jobs WHERE id=?1 AND state IN ('queued','running'))",[&parent],|row|row.get(0))?;
                anyhow::ensure!(!running,"Stop the parent video before deleting recovery inputs");
                let mut stmt=conn.prepare("SELECT owned.generation_id,q.state FROM chain_stage_generations owned JOIN generation_queue q ON q.id=owned.generation_id WHERE owned.video_parent_id=?1 AND owned.generation_id>?2 ORDER BY owned.generation_id LIMIT 50")?;
                let rows=stmt.query_map((&parent,&after),|row|Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?)))?.collect::<Result<Vec<_>, _>>()?;
                anyhow::ensure!(rows.iter().all(|(_,state)|state!="running"),"A video clip is still running. Wait for Stop before deleting recovery inputs.");
                Ok(rows.into_iter().map(|(id,_)|id).collect())
            })
        }).await.map_err(|e|ApiError::internal(format!("stage cleanup stopped: {e}")))?.map_err(|e|ApiError::with_code(e.to_string(),"VIDEO_CHILD_STILL_RUNNING",StatusCode::CONFLICT))?;
        if page.is_empty() {
            return Ok(());
        }
        cursor = page.last().expect("nonempty stage page").clone();
        let _transition = state.queue_journal.lock_durable_transition().await;
        for id in page {
            match crate::routes::cancel_one_queue_job(state, &id).await? {
                crate::routes::QueueJobCancelOutcome::Pending
                | crate::routes::QueueJobCancelOutcome::Completing => {
                    return Err(ApiError::with_code(
                        "A video clip is still stopping. Try deleting again after it returns.",
                        "VIDEO_CHILD_STILL_RUNNING",
                        StatusCode::CONFLICT,
                    ))
                }
                crate::routes::QueueJobCancelOutcome::NotFound => {
                    let db = state.metadata_db.clone();
                    let probe = id.clone();
                    let remains = tokio::task::spawn_blocking(move || {
                        db.as_ref()
                            .as_ref()
                            .map(|db| mold_db::generation_queue::get(db, &probe))
                            .transpose()
                            .map(|row| row.flatten().is_some())
                    })
                    .await
                    .map_err(|e| ApiError::internal(format!("stage cleanup check stopped: {e}")))?
                    .map_err(|e| {
                        ApiError::internal(format!("stage cleanup check failed: {e:#}"))
                    })?;
                    if remains {
                        return Err(ApiError::with_code("This clip's retained queue ownership could not be released. Reconnect to its original machine before deleting.","VIDEO_CHILD_RECOVERY_UNAVAILABLE",StatusCode::CONFLICT));
                    }
                }
                _ => {}
            }
        }
    }
}

pub(crate) async fn require_no_owned_waiting_generations(
    state: &AppState,
    paused_only: bool,
) -> Result<(), ApiError> {
    let db = state.metadata_db.clone();
    let Some(owner) = state.queue_journal.owner_uuid().map(str::to_string) else {
        return Ok(());
    };
    tokio::task::spawn_blocking(move || {
        if let Some(db)=db.as_ref().as_ref() {
            let has_owned=db.with_conn(|conn| {
                conn.query_row("SELECT EXISTS(SELECT 1 FROM chain_stage_generations owned JOIN generation_queue q ON q.id=owned.generation_id WHERE q.owner_uuid=?2 AND ((?1 AND q.state='paused') OR (NOT ?1 AND q.state IN ('queued','paused')))) OR (?1 AND EXISTS(SELECT 1 FROM video_job_child_owners owned JOIN chain_jobs c ON c.id=owned.child_id WHERE c.state='paused'))",(paused_only,owner),|row|row.get::<_,bool>(0)).map_err(Into::into)
            }).map_err(|e|ApiError::internal(format!("could not inspect video ownership: {e:#}")))?;
            if has_owned {return Err(owned())}
        }
        Ok(())
    }).await.map_err(|e|ApiError::internal(format!("ownership check stopped: {e}")))?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn reserved_child_controls_refuse_even_when_parent_and_chain_are_missing() {
        let mut state = AppState::for_tests();
        state.metadata_db =
            std::sync::Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap()));
        let home = tempfile::tempdir().unwrap();
        state.queue_journal = std::sync::Arc::new(crate::queue_journal::QueueJournal::new(
            state.metadata_db.clone(),
            Some(home.path()),
            "owner-test",
        ));
        let db = state.metadata_db.as_ref().as_ref().unwrap();
        db.with_conn(|conn| {
            conn.execute("INSERT INTO video_job_child_owners(child_id,parent_id) VALUES('chain','removed-parent')",[])?;
            conn.execute("INSERT INTO generation_batches(id,client_batch_id,owner_uuid,request_sha256,created_at_ms) VALUES('batch','client','owner','receipt',0)",[])?;
            conn.execute("INSERT INTO generation_batch_children(batch_id,job_id,batch_index,state,updated_at_ms) VALUES('batch','stage',1,'accepted',0)",[])?;
            conn.execute("INSERT INTO chain_stage_generations(generation_id,chain_id,stage_index,video_parent_id) VALUES('stage','chain',0,'removed-parent')",[])?;
            Ok(())
        }).unwrap();
        assert_eq!(
            require_public_chain(db, "chain").unwrap_err().code,
            "VIDEO_JOB_CHILD_OWNED"
        );
        assert!(require_public_chain(db, "ordinary-chain").is_ok());
        assert_eq!(
            require_public_generation(&state, "stage")
                .await
                .unwrap_err()
                .code,
            "VIDEO_JOB_CHILD_OWNED"
        );
        assert!(require_public_generation(&state, "ordinary-stage")
            .await
            .is_ok());
        assert!(require_no_owned_waiting_generations(&state, false)
            .await
            .is_ok());
        db.with_conn(|conn| {conn.execute("INSERT INTO generation_queue(id,owner_uuid,state,model,request_json,output_dir,completion_payload,created_at,updated_at) VALUES('stage','owner','paused','mock','{}','/gallery','metadata_only',0,0)",[])?;Ok(())}).unwrap();
        db.with_conn(|conn| {
            conn.execute(
                "UPDATE generation_queue SET owner_uuid=?1",
                [state.queue_journal.owner_uuid().unwrap()],
            )?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            require_no_owned_waiting_generations(&state, false)
                .await
                .unwrap_err()
                .code,
            "VIDEO_JOB_CHILD_OWNED"
        );
        assert_eq!(
            require_no_owned_waiting_generations(&state, true)
                .await
                .unwrap_err()
                .code,
            "VIDEO_JOB_CHILD_OWNED"
        );
        assert!(mold_db::generation_queue::resume_all_paused(
            db,
            state.queue_journal.owner_uuid().unwrap(),
            1
        )
        .is_err());
        assert_eq!(
            mold_db::generation_queue::get(db, "stage")
                .unwrap()
                .unwrap()
                .state,
            mold_db::generation_queue::QueueRowState::Paused
        );
    }
}
