//! Reads deterministic local fixture configuration for protocol UAT.
use mold_secure_client::{PairedHost, SecureClient};
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let file = std::env::args_os()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("provide owner-only fixture configuration file"))?;
    let private = mold_secure_client::read_private(std::path::Path::new(&file))?;
    let host: PairedHost = if std::env::args().any(|a| a == "--pair") {
        let invite = serde_json::from_slice(&private)?;
        let host = mold_secure_client::pair(invite, "Rust fixture probe").await?;
        println!("Invitation claimed through encrypted channel");
        host
    } else {
        serde_json::from_slice(&private)?
    };
    let origin = host.origin.clone();
    let client = SecureClient::new(host)?;
    if let Some(count) = std::env::var("MOLD_SECURE_PROBE_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        let start = std::time::Instant::now();
        let mut response = client
            .send(
                reqwest::Client::new()
                    .get(format!("{origin}/bulk?bytes={count}"))
                    .build()?,
            )
            .await?;
        let mut read = 0;
        while let Some(chunk) = response.chunk().await? {
            anyhow::ensure!(chunk.iter().all(|b| *b == 42), "bulk integrity mismatch");
            read += chunk.len();
            if read % (16 * 1024 * 1024) < chunk.len() {
                println!(
                    "bulk {} MiB, {}ms",
                    read / 1024 / 1024,
                    start.elapsed().as_millis()
                );
            }
        }
        anyhow::ensure!(read == count, "bulk length mismatch");
        println!(
            "BULK passed {} bytes, {} ms, {:.3} MiB/s",
            read,
            start.elapsed().as_millis(),
            read as f64 / 1048576.0 / start.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    let overall = std::time::Instant::now();
    for (method, path, body) in [
        ("GET", "/api/status", vec![]),
        (
            "POST",
            "/echo",
            vec![
                37;
                std::env::var("MOLD_SECURE_PROBE_UPLOAD_BYTES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1048576)
            ],
        ),
        ("GET", "/range", vec![]),
        ("GET", "/sse", vec![]),
    ] {
        let started = std::time::Instant::now();
        let request = reqwest::Client::new()
            .request(method.parse()?, format!("{origin}{path}"))
            .body(body.clone())
            .build()?;
        let response = client.send(request).await?;
        let status = response.status();
        let result = response.bytes().await?;
        if method == "POST" {
            anyhow::ensure!(result == body, "echo differs")
        };
        println!(
            "{method} {path}: {status}, {} bytes, {} ms",
            result.len(),
            started.elapsed().as_millis()
        );
    }
    let results = futures_util::future::join_all((0..8).map(|_| {
        let client = client.clone();
        let origin = origin.clone();
        async move {
            let response = client
                .send(
                    reqwest::Client::new()
                        .get(format!("{origin}/range"))
                        .build()?,
                )
                .await?;
            anyhow::ensure!(response.bytes().await?.len() == 196608, "range mismatch");
            Ok::<_, anyhow::Error>(())
        }
    }))
    .await;
    for result in results {
        result?;
    }
    println!(
        "8 concurrent encrypted range requests passed; total {} ms",
        overall.elapsed().as_millis()
    );
    Ok(())
}
