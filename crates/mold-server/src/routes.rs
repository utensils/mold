use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use mold_core::{GenerateRequest, GpuInfo, ModelInfo, OutputFormat, ServerStatus};
use mold_inference::model_registry;

use crate::state::AppState;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/generate", post(generate))
        .route("/api/models", get(list_models))
        .route("/api/status", get(server_status))
        .route("/health", get(health))
        .with_state(state)
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Validate request before touching the engine
    if let Err(e) = validate_generate_request(&req) {
        return Err((StatusCode::UNPROCESSABLE_ENTITY, e));
    }

    // Load on first request (holds lock only during load, not inference)
    {
        let mut engine = state.engine.lock().await;
        if !engine.is_loaded() {
            tracing::info!("first request — loading model...");
            engine.load().map_err(|e| {
                tracing::error!("model load failed: {e:#}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("model load error: {e}"),
                )
            })?;
        }
    }

    // Run inference in a blocking task to avoid starving the async executor.
    // Using Arc<Mutex> so the lock is held only during the blocking work, and
    // panics are caught and converted to proper error responses.
    let engine = state.engine.clone();
    let result = tokio::task::spawn_blocking(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = engine.blocking_lock();
            guard.generate(&req)
        }))
    })
    .await
    .map_err(|e| {
        tracing::error!("inference task join error: {e:?}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference task failed".to_string(),
        )
    })?;

    let mut response = match result {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => {
            tracing::error!("generation error: {e:#}");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("generation error: {e}"),
            ));
        }
        Err(panic_payload) => {
            let msg = panic_payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            tracing::error!("inference panicked: {msg}");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("inference panicked: {msg}"),
            ));
        }
    };

    // Return raw image bytes with correct Content-Type
    let img = response.images.remove(0);
    let content_type = match img.format {
        OutputFormat::Png => "image/png",
        OutputFormat::Jpeg => "image/jpeg",
    };
    let headers = [(header::CONTENT_TYPE, content_type)];
    Ok((headers, img.data))
}

fn validate_generate_request(req: &GenerateRequest) -> Result<(), String> {
    if req.prompt.trim().is_empty() {
        return Err("prompt must not be empty".to_string());
    }
    if req.width == 0 || req.height == 0 {
        return Err("width and height must be > 0".to_string());
    }
    if req.width % 16 != 0 || req.height % 16 != 0 {
        return Err(format!(
            "width ({}) and height ({}) must be multiples of 16 (FLUX patchification requirement)",
            req.width, req.height
        ));
    }
    if req.width > 1024 || req.height > 1024 {
        return Err(format!(
            "width ({}) and height ({}) must be <= 1024",
            req.width, req.height
        ));
    }
    if req.steps == 0 {
        return Err("steps must be >= 1".to_string());
    }
    if req.steps > 100 {
        return Err(format!("steps ({}) must be <= 100", req.steps));
    }
    Ok(())
}

async fn list_models(State(state): State<AppState>) -> Json<Vec<ModelInfo>> {
    let engine = state.engine.lock().await;
    let loaded_name = engine.model_name().to_string();
    let is_loaded = engine.is_loaded();

    let models: Vec<ModelInfo> = model_registry::known_models()
        .into_iter()
        .map(|mut m| {
            m.is_loaded = is_loaded && m.name == loaded_name;
            m
        })
        .collect();

    Json(models)
}

async fn server_status(State(state): State<AppState>) -> Json<ServerStatus> {
    let engine = state.engine.lock().await;

    let models_loaded = if engine.is_loaded() {
        vec![engine.model_name().to_string()]
    } else {
        vec![]
    };

    Json(ServerStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        models_loaded,
        gpu_info: query_gpu_info(),
        uptime_secs: state.start_time.elapsed().as_secs(),
    })
}

/// Query GPU info via nvidia-smi. Returns None if not available or on non-NVIDIA hardware.
fn query_gpu_info() -> Option<GpuInfo> {
    // Try PATH first, then NixOS well-known location.
    let nvidia_smi = ["nvidia-smi", "/run/current-system/sw/bin/nvidia-smi"]
        .iter()
        .find(|p| {
            if p.starts_with('/') {
                std::path::Path::new(p).exists()
            } else {
                true // let Command::new try PATH
            }
        })
        .copied()
        .unwrap_or("nvidia-smi");

    let output = std::process::Command::new(nvidia_smi)
        .args([
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8(output.stdout).ok()?;
    let line = text.lines().next()?;
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 3 {
        return None;
    }

    let name = parts[0].to_string();
    let vram_total_mb = parts[1].parse::<u64>().ok()?;
    let vram_used_mb = parts[2].parse::<u64>().ok()?;

    Some(GpuInfo {
        name,
        vram_total_mb,
        vram_used_mb,
    })
}

async fn health() -> impl IntoResponse {
    StatusCode::OK
}
