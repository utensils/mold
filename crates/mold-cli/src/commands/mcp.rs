use anyhow::Result;
use base64::{engine::general_purpose, Engine as _};
use mold_core::{Config, GenerateRequest, MoldClient, OutputFormat};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};

const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

pub async fn run(host: Option<String>) -> Result<()> {
    let server = McpServer::new(host);
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = BufWriter::new(tokio::io::stdout());

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Value>(&line) {
            Ok(message) => server.handle_message(message).await,
            Err(e) => Some(error_response(
                Value::Null,
                -32700,
                format!("parse error: {e}"),
            )),
        };

        if let Some(response) = response {
            let encoded = serde_json::to_string(&response)?;
            stdout.write_all(encoded.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}

struct McpServer {
    client: MoldClient,
}

impl McpServer {
    fn new(host: Option<String>) -> Self {
        let client = match host {
            Some(host) => match std::env::var("MOLD_API_KEY").ok().filter(|k| !k.is_empty()) {
                Some(api_key) => MoldClient::with_api_key(&host, api_key),
                None => MoldClient::new(&host),
            },
            None => MoldClient::from_env(),
        };
        Self { client }
    }

    async fn handle_message(&self, message: Value) -> Option<Value> {
        match message {
            Value::Array(items) => {
                if items.is_empty() {
                    return Some(error_response(
                        Value::Null,
                        -32600,
                        "invalid request: empty batch",
                    ));
                }
                let mut responses = Vec::new();
                for item in items {
                    if let Some(response) = self.handle_single(item).await {
                        responses.push(response);
                    }
                }
                (!responses.is_empty()).then_some(Value::Array(responses))
            }
            other => self.handle_single(other).await,
        }
    }

    async fn handle_single(&self, message: Value) -> Option<Value> {
        let method = match message.get("method").and_then(Value::as_str) {
            Some(method) => method,
            None => {
                return Some(error_response(
                    message.get("id").cloned().unwrap_or(Value::Null),
                    -32600,
                    "invalid request: missing method",
                ));
            }
        };

        if method == "tools/call" {
            return self.handle_tool_call(&message).await;
        }

        handle_protocol_message(message)
    }

    async fn handle_tool_call(&self, message: &Value) -> Option<Value> {
        let id = message.get("id").cloned()?;

        let params = message.get("params").cloned().unwrap_or_else(|| json!({}));
        let Some(name) = params.get("name").and_then(Value::as_str) else {
            return Some(error_response(id, -32602, "tools/call missing params.name"));
        };
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));

        let result = match name {
            "generate_image" => self.tool_generate_image(arguments).await,
            "list_models" => self.tool_list_models(arguments).await,
            "server_status" => self.tool_server_status().await,
            other => Err(format!("unknown tool: {other}")),
        };

        Some(match result {
            Ok(result) => response(id, result),
            Err(err) => response(
                id,
                json!({
                    "content": [{ "type": "text", "text": err }],
                    "isError": true
                }),
            ),
        })
    }

    async fn tool_generate_image(&self, arguments: Value) -> std::result::Result<Value, String> {
        let args: GenerateImageArgs =
            serde_json::from_value(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
        let req = build_generate_request(args)?;
        let response = self
            .client
            .generate(req)
            .await
            .map_err(|e| format!("mold generation failed: {e}"))?;
        let image = response
            .images
            .first()
            .ok_or_else(|| "mold did not return an image".to_string())?;
        let encoded = general_purpose::STANDARD.encode(&image.data);
        let mut details = format!(
            "Generated {}x{} {} image with {} in {:.1}s; seed {}",
            image.width,
            image.height,
            image.format.extension(),
            response.model,
            response.generation_time_ms as f64 / 1000.0,
            response.seed_used
        );
        if let Some(gpu) = response.gpu {
            details.push_str(&format!("; gpu {gpu}"));
        }

        Ok(json!({
            "content": [
                { "type": "text", "text": details },
                {
                    "type": "image",
                    "data": encoded,
                    "mimeType": image.format.content_type()
                }
            ]
        }))
    }

    async fn tool_list_models(&self, arguments: Value) -> std::result::Result<Value, String> {
        let args: ListModelsArgs =
            serde_json::from_value(arguments).map_err(|e| format!("invalid arguments: {e}"))?;
        let models = self
            .client
            .list_models_extended()
            .await
            .map_err(|e| format!("failed to list models: {e}"))?;

        let limit = args.limit.unwrap_or(50).clamp(1, 200);
        let mut lines = Vec::new();
        for model in models
            .into_iter()
            .filter(|m| !args.downloaded_only.unwrap_or(true) || m.downloaded)
            .filter(|m| !args.generation_only.unwrap_or(true) || m.is_generation_model())
            .take(limit)
        {
            let status = if model.downloaded {
                "downloaded"
            } else {
                "not downloaded"
            };
            lines.push(format!(
                "- {} [{}] {status}; default {}x{}, steps {}, guidance {}",
                model.name,
                model.family,
                model.defaults.default_width,
                model.defaults.default_height,
                model.defaults.default_steps,
                model.defaults.default_guidance
            ));
        }

        if lines.is_empty() {
            lines.push("No matching models found.".to_string());
        }

        Ok(text_result(lines.join("\n")))
    }

    async fn tool_server_status(&self) -> std::result::Result<Value, String> {
        let status = self
            .client
            .server_status()
            .await
            .map_err(|e| format!("failed to read server status: {e}"))?;

        let mut lines = vec![
            format!("mold server {}", status.version),
            format!("busy: {}", status.busy),
            format!("loaded models: {}", status.models_loaded.join(", ")),
            format!("uptime: {}s", status.uptime_secs),
        ];
        if let Some(hostname) = status.hostname {
            lines.push(format!("host: {hostname}"));
        }
        if let Some(memory) = status.memory_status {
            lines.push(memory);
        }
        if let Some(gpus) = status.gpus {
            for gpu in gpus {
                lines.push(format!(
                    "gpu {}: {} ({:?}){}",
                    gpu.ordinal,
                    gpu.name,
                    gpu.state,
                    gpu.loaded_model
                        .map(|m| format!(", loaded {m}"))
                        .unwrap_or_default()
                ));
            }
        }
        if let Some(depth) = status.queue_depth {
            lines.push(format!("queue depth: {depth}"));
        }

        Ok(text_result(lines.join("\n")))
    }
}

#[derive(Debug, Deserialize)]
struct GenerateImageArgs {
    prompt: String,
    model: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f64>,
    seed: Option<u64>,
    negative_prompt: Option<String>,
    output_format: Option<String>,
    expand: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ListModelsArgs {
    downloaded_only: Option<bool>,
    generation_only: Option<bool>,
    limit: Option<usize>,
}

fn build_generate_request(args: GenerateImageArgs) -> std::result::Result<GenerateRequest, String> {
    if args.prompt.trim().is_empty() {
        return Err("prompt must not be empty".to_string());
    }

    let output_format = match args.output_format.as_deref().unwrap_or("png") {
        "png" => OutputFormat::Png,
        "jpeg" | "jpg" => OutputFormat::Jpeg,
        other => {
            return Err(format!(
                "unsupported output_format '{other}'; use png or jpeg"
            ))
        }
    };

    let config = Config::load_or_default();
    let model = args
        .model
        .unwrap_or_else(|| config.resolved_default_model());
    let model_cfg = config.resolved_model_config(&model);
    let width = args
        .width
        .unwrap_or_else(|| model_cfg.effective_width(&config));
    let height = args
        .height
        .unwrap_or_else(|| model_cfg.effective_height(&config));

    if width == 0 || height == 0 {
        return Err("width and height must be greater than zero".to_string());
    }
    if width & 15 != 0 || height & 15 != 0 {
        return Err("width and height must be multiples of 16".to_string());
    }

    Ok(GenerateRequest {
        prompt: args.prompt,
        negative_prompt: args.negative_prompt,
        model,
        width,
        height,
        steps: args
            .steps
            .unwrap_or_else(|| model_cfg.effective_steps(&config)),
        guidance: args
            .guidance
            .unwrap_or_else(|| model_cfg.effective_guidance()),
        seed: args.seed,
        batch_size: 1,
        output_format: Some(output_format),
        embed_metadata: Some(config.effective_embed_metadata(None)),
        scheduler: None,
        cfg_plus: None,
        source_image: None,
        edit_images: None,
        strength: 0.75,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: args.expand,
        original_prompt: None,
        lora: None,
        frames: None,
        fps: None,
        upscale_model: None,
        gif_preview: false,
        enable_audio: None,
        audio_file: None,
        source_video: None,
        keyframes: None,
        pipeline: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: None,
    })
}

fn handle_protocol_message(message: Value) -> Option<Value> {
    let id = message.get("id").cloned();
    let method = message.get("method").and_then(Value::as_str)?;

    match (method, id) {
        ("notifications/initialized", _) => None,
        (_, None) => None,
        ("initialize", Some(id)) => Some(response(
            id,
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "mold",
                    "version": mold_core::build_info::version_string()
                }
            }),
        )),
        ("ping", Some(id)) => Some(response(id, json!({}))),
        ("tools/list", Some(id)) => Some(response(id, json!({ "tools": tool_definitions() }))),
        ("prompts/list", Some(id)) => Some(response(id, json!({ "prompts": [] }))),
        ("resources/list", Some(id)) => Some(response(id, json!({ "resources": [] }))),
        (other, Some(id)) => Some(error_response(
            id,
            -32601,
            format!("method not found: {other}"),
        )),
    }
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "generate_image",
            "description": "Generate one image with mold. Requires a running mold serve process, unless MOLD_HOST points at a remote mold server.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The image prompt."
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional mold model name, such as flux2-klein:q8. Defaults to mold's configured default model."
                    },
                    "width": {
                        "type": "integer",
                        "minimum": 16,
                        "multipleOf": 16,
                        "description": "Image width in pixels."
                    },
                    "height": {
                        "type": "integer",
                        "minimum": 16,
                        "multipleOf": 16,
                        "description": "Image height in pixels."
                    },
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Inference step count."
                    },
                    "guidance": {
                        "type": "number",
                        "description": "Guidance scale."
                    },
                    "seed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional deterministic seed."
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Optional negative prompt for CFG-capable model families."
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["png", "jpeg", "jpg"],
                        "description": "Output image format. Defaults to png."
                    },
                    "expand": {
                        "type": "boolean",
                        "description": "Ask the mold server to expand the prompt before generation."
                    }
                },
                "required": ["prompt"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_models",
            "description": "List mold models visible to the running mold server.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "downloaded_only": {
                        "type": "boolean",
                        "description": "Only show downloaded models. Defaults to true."
                    },
                    "generation_only": {
                        "type": "boolean",
                        "description": "Hide upscalers, utility models, and auxiliary models. Defaults to true."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Maximum number of models to return. Defaults to 50."
                    }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "server_status",
            "description": "Show mold server health, queue, loaded model, and GPU status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        }
    ])
}

fn response(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

fn error_response(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into()
        }
    })
}

fn text_result(text: impl Into<String>) -> Value {
    json!({
        "content": [{ "type": "text", "text": text.into() }]
    })
}

#[cfg(test)]
fn handle_protocol_message_for_test(message: Value) -> Option<Value> {
    handle_protocol_message(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn initialize_declares_tools_capability() {
        let response = handle_protocol_message_for_test(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": { "protocolVersion": "2025-06-18" }
        }))
        .expect("initialize should produce a response");

        assert_eq!(response["id"], 1);
        assert_eq!(response["result"]["capabilities"]["tools"], json!({}));
        assert_eq!(response["result"]["serverInfo"]["name"], "mold");
    }

    #[test]
    fn tools_list_exposes_generate_image() {
        let response = handle_protocol_message_for_test(json!({
            "jsonrpc": "2.0",
            "id": "tools",
            "method": "tools/list"
        }))
        .expect("tools/list should produce a response");

        let tools = response["result"]["tools"].as_array().unwrap();
        assert!(tools.iter().any(|tool| tool["name"] == "generate_image"));
    }
}
