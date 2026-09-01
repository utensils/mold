use anyhow::Result;
use mold_core::{
    CreateVideoUpscaleJobRequest, MoldClient, VideoUpscaleJobState, VideoUpscaleSource,
    VIDEO_UPSCALE_DISCLOSURE,
};

fn client(host: Option<String>) -> MoldClient {
    MoldClient::new(
        host.or_else(|| std::env::var("MOLD_HOST").ok())
            .as_deref()
            .unwrap_or("http://localhost:7680"),
    )
}

pub async fn create(
    filename: String,
    model: String,
    tile_size: Option<u32>,
    host: Option<String>,
    wait: bool,
) -> Result<()> {
    let client = client(host);
    let mut job = client
        .create_video_upscale_job(&CreateVideoUpscaleJobRequest {
            source: VideoUpscaleSource::Library { filename },
            model,
            tile_size,
        })
        .await?;
    eprintln!("{VIDEO_UPSCALE_DISCLOSURE}");
    println!("{}", job.id);
    if wait {
        while !job.state.is_terminal() {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            job = client.get_video_upscale_job(&job.id).await?;
            eprintln!(
                "{:?}: {}/{} frames",
                job.state, job.completed_frames, job.total_frames
            );
        }
        if job.state != VideoUpscaleJobState::Completed {
            anyhow::bail!(job
                .error
                .unwrap_or_else(|| format!("job ended {:?}", job.state)));
        }
        if let Some(filename) = job.output_filename {
            println!("{filename}");
        }
    }
    Ok(())
}

pub async fn list(host: Option<String>) -> Result<()> {
    for job in client(host).list_video_upscale_jobs().await? {
        println!(
            "{}\t{:?}\t{}/{}\t{}",
            job.id,
            job.state,
            job.completed_frames,
            job.total_frames,
            job.output_filename.unwrap_or_default()
        );
    }
    Ok(())
}

pub async fn status(id: String, host: Option<String>) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&client(host).get_video_upscale_job(&id).await?)?
    );
    Ok(())
}

pub async fn transition(id: String, action: &'static str, host: Option<String>) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(
            &client(host)
                .transition_video_upscale_job(&id, action)
                .await?
        )?
    );
    Ok(())
}
