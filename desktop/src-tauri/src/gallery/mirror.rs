use super::*;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

static MIRROR_PERMITS: Semaphore = Semaphore::const_new(1);

const MAX_TRANSFER_BYTES: u64 = 512 * 1024 * 1024;
const MAX_DESCRIPTOR_BYTES: usize = 64 * 1024;

#[derive(Clone, Deserialize, Serialize)]
struct Member {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    member_id: Option<String>,
    role: String,
    position: String,
    sink: String,
    size_bytes: u64,
    sha256: String,
}
#[derive(Clone, Deserialize, Serialize)]
struct Offer {
    archive_identity_sha256: String,
    members: Vec<Member>,
    output_sha256: Option<String>,
    output_size_bytes: Option<u64>,
    metadata: Option<Box<mold_core::OutputMetadata>>,
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}
fn has_source_markers(metadata: &mold_core::OutputMetadata) -> bool {
    metadata.source_image_sha256.is_some()
        || metadata.id_image_sha256.is_some()
        || metadata.source_video_path.is_some()
        || metadata.audio_file_path.is_some()
        || metadata.extend_video_path.is_some()
        || metadata.extend_overlap_frames.is_some()
        || metadata
            .edit_image_sha256s
            .as_ref()
            .is_some_and(|items| !items.is_empty())
        || metadata
            .id_image_sha256s
            .as_ref()
            .is_some_and(|items| !items.is_empty())
        || metadata
            .references
            .as_ref()
            .is_some_and(|items| !items.is_empty())
        || metadata
            .keyframes
            .as_ref()
            .is_some_and(|items| !items.is_empty())
}
fn valid_slot(member: &Member) -> bool {
    let indexed = member.position.strip_prefix("item:").is_some_and(|value| {
        value
            .parse::<u32>()
            .is_ok_and(|number| number.to_string() == value)
    });
    match member.role.as_str() {
        "identity_images" | "edit_images" | "references" | "keyframes" => indexed,
        "matting_processed_references" => matches!(
            member.position.as_str(),
            "front" | "left" | "back" | "right"
        ),
        "source_image"
        | "identity_image"
        | "mask_image"
        | "control_image"
        | "audio_file"
        | "audio_file_path"
        | "source_video"
        | "source_video_path"
        | "extend_video"
        | "extend_video_path"
        | "matting_processed_source_image" => member.position == "scalar",
        _ => false,
    }
}
fn validate(offer: &Offer) -> Result<(), String> {
    if offer.members.len() > 64 || !valid_digest(&offer.archive_identity_sha256) {
        return Err("Invalid retained source-media offer.".into());
    }
    let mut seen = std::collections::HashSet::new();
    let total = offer.members.iter().try_fold(0_u64, |total, member| {
        if member.member_id.as_deref().is_none_or(str::is_empty)
            || !valid_slot(member)
            || !valid_digest(&member.sha256)
            || member.size_bytes == 0
            || !seen.insert((&member.role, &member.position))
            || !matches!(member.sink.as_str(), "memory" | "private_staging")
        {
            return None;
        }
        total.checked_add(member.size_bytes)
    });
    if total.is_none_or(|total| total > MAX_TRANSFER_BYTES) {
        return Err("The retained source media is too large or incomplete to copy.".into());
    }
    Ok(())
}
fn same_output(left: &Offer, right: &Offer) -> Result<(), String> {
    if left.output_sha256.is_none()
        || left.output_size_bytes.is_none()
        || left.metadata.is_none()
        || left.output_sha256 != right.output_sha256
        || left.output_size_bytes != right.output_size_bytes
        || serde_json::to_value(&left.metadata).ok() != serde_json::to_value(&right.metadata).ok()
    {
        return Err("The print changed during copying. Try again.".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_transfer_limits_and_snapshot_checks() {
        let mut offer = Offer {
            archive_identity_sha256: "a".repeat(64),
            members: vec![],
            output_sha256: Some("b".repeat(64)),
            output_size_bytes: Some(3),
            metadata: None,
        };
        assert!(validate(&offer).is_ok());
        let member = Member {
            member_id: Some("m".into()),
            role: "source_image".into(),
            position: "scalar".into(),
            sink: "memory".into(),
            size_bytes: MAX_TRANSFER_BYTES + 1,
            sha256: "c".repeat(64),
        };
        offer.members.push(member.clone());
        assert!(validate(&offer).is_err());
        offer.members = vec![
            Member {
                size_bytes: 1,
                ..member
            };
            65
        ];
        assert!(validate(&offer).is_err());
        offer.members = vec![
            Member {
                size_bytes: 1,
                ..offer.members[0].clone()
            };
            2
        ];
        assert!(
            validate(&offer).is_err(),
            "duplicate slots cannot be imported"
        );
        assert!(
            same_output(&offer, &offer).is_err(),
            "metadata is mandatory"
        );
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
    #[test]
    fn copied_output_must_match_bytes_size_and_authoritative_settings() {
        let request = serde_json::from_value::<mold_core::GenerateRequest>(serde_json::json!({
            "prompt": "original", "model": "flux-dev:q4", "width": 8, "height": 8,
            "steps": 4, "guidance": 1.0, "batch_size": 1
        }))
        .unwrap();
        let offer = Offer {
            archive_identity_sha256: "a".repeat(64),
            members: vec![],
            output_sha256: Some(digest(b"abc")),
            output_size_bytes: Some(3),
            metadata: Some(Box::new(mold_core::OutputMetadata::from_generate_request(
                &request, 1, None, "test",
            ))),
        };
        assert!(same_output(&offer, &offer).is_ok());
        let mut changed = offer.clone();
        changed.output_sha256 = Some(digest(b"changed"));
        assert!(same_output(&offer, &changed).is_err());
        changed = offer.clone();
        changed.output_size_bytes = Some(4);
        assert!(same_output(&offer, &changed).is_err());
        changed = offer.clone();
        changed.metadata.as_mut().unwrap().prompt = "different settings".into();
        assert!(same_output(&offer, &changed).is_err());
    }

    fn fake_host(
        responses: Vec<(&'static str, &'static str)>,
    ) -> (MediaSaveTarget, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            for (status, body) in responses {
                let (mut connection, _) = listener.accept().unwrap();
                let mut request = [0; 4096];
                let _received = std::io::Read::read(&mut connection, &mut request).unwrap();
                std::io::Write::write_all(&mut connection, format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
            }
        });
        (
            MediaSaveTarget {
                base_url: format!("http://{address}"),
                api_key: None,
            },
            worker,
        )
    }

    #[tokio::test]
    async fn old_transfer_route_only_falls_back_for_verified_legacy_archive() {
        for (availability, allowed) in [
            (r#"{"availability":"unavailable_legacy"}"#, true),
            (r#"{"availability":"available","members":[]}"#, false),
            (r#"{"availability":"unavailable_auth"}"#, false),
            (
                r#"{"availability":"unavailable_missing_or_corrupt"}"#,
                false,
            ),
        ] {
            let (host, worker) = fake_host(vec![("404 Not Found", "{}"), ("200 OK", availability)]);
            let result = offer(
                &reqwest::Client::new(),
                &host,
                "/api/gallery/source-media/a.png",
                true,
            )
            .await;
            assert_eq!(result.is_ok(), allowed);
            worker.join().unwrap();
        }
    }

    #[tokio::test]
    async fn transfer_response_is_bounded_before_body_allocation() {
        let (host, worker) = fake_host(vec![("200 OK", "abcd")]);
        let response = reqwest::Client::new()
            .get(host.base_url)
            .send()
            .await
            .unwrap();
        assert!(bounded_response(response, 3).await.is_err());
        worker.join().unwrap();
    }
    fn read_http(stream: &mut std::net::TcpStream) -> (String, Vec<u8>) {
        use std::io::BufRead;
        let mut reader = std::io::BufReader::new(stream);
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            assert!(reader.read_line(&mut line).unwrap() > 0);
            headers.push_str(&line);
            if line == "\r\n" {
                break;
            }
        }
        let mut body = vec![];
        if headers
            .to_lowercase()
            .contains("transfer-encoding: chunked")
        {
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let size =
                    usize::from_str_radix(line.trim().split(';').next().unwrap(), 16).unwrap();
                if size == 0 {
                    break;
                }
                let start = body.len();
                body.resize(start + size, 0);
                std::io::Read::read_exact(&mut reader, &mut body[start..]).unwrap();
                let mut newline = [0; 2];
                std::io::Read::read_exact(&mut reader, &mut newline).unwrap();
            }
        } else if let Some(length) = headers.lines().find_map(|line| {
            line.to_lowercase()
                .strip_prefix("content-length:")
                .map(|value| value.trim().parse::<usize>().unwrap())
        }) {
            body.resize(length, 0);
            std::io::Read::read_exact(&mut reader, &mut body).unwrap();
        }
        (headers, body)
    }
    fn protocol_host(
        count: usize,
        mut reply: impl FnMut(&str, &[u8]) -> Vec<u8> + Send + 'static,
    ) -> (MediaSaveTarget, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            for _ in 0..count {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                let (headers, body) = read_http(&mut stream);
                let result = reply(&headers, &body);
                std::io::Write::write_all(
                    &mut stream,
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        result.len()
                    )
                    .as_bytes(),
                )
                .unwrap();
                std::io::Write::write_all(&mut stream, &result).unwrap();
            }
        });
        (
            MediaSaveTarget {
                base_url: format!("http://{address}"),
                api_key: None,
            },
            worker,
        )
    }

    #[tokio::test]
    async fn two_hosts_copy_output_and_private_sources_to_actual_imported_filename() {
        let output = b"output bytes".to_vec();
        let private = b"private source bytes".to_vec();
        let request = serde_json::from_value::<mold_core::GenerateRequest>(serde_json::json!({
            "prompt": "test", "model": "flux-dev:q4", "width": 8, "height": 8,
            "steps": 4, "guidance": 1.0, "batch_size": 1
        }))
        .unwrap();
        let metadata = Box::new(mold_core::OutputMetadata::from_generate_request(
            &request, 1, None, "test",
        ));
        let origin_offer = Offer {
            archive_identity_sha256: "a".repeat(64),
            members: vec![Member {
                member_id: Some("private-member".into()),
                role: "source_image".into(),
                position: "scalar".into(),
                sink: "memory".into(),
                size_bytes: private.len() as u64,
                sha256: digest(&private),
            }],
            output_sha256: Some(digest(&output)),
            output_size_bytes: Some(output.len() as u64),
            metadata: Some(metadata.clone()),
        };
        let offer_bytes = serde_json::to_vec(&origin_offer).unwrap();
        let source_output = output.clone();
        let source_private = private.clone();
        let (mut source, source_thread) = protocol_host(4, move |headers, _| {
            assert!(headers.to_lowercase().contains("x-api-key: source-key"));
            if headers.starts_with("GET /api/gallery/image/") {
                return source_output.clone();
            }
            if headers.lines().next().unwrap().contains("/transfer ") {
                return offer_bytes.clone();
            }
            assert!(headers
                .lines()
                .next()
                .unwrap()
                .contains("/private%2Dmember "));
            source_private.clone()
        });
        source.api_key = Some("source-key".into());
        let destination_offer = Offer {
            archive_identity_sha256: "b".repeat(64),
            members: vec![],
            ..origin_offer.clone()
        };
        let destination_offer_bytes = serde_json::to_vec(&destination_offer).unwrap();
        let destination_identity = destination_offer.archive_identity_sha256.clone();
        let expected_metadata = serde_json::to_value(&metadata).unwrap();
        let (destination, destination_thread) = protocol_host(3, move |headers, body| {
            assert!(headers
                .to_lowercase()
                .contains("x-api-key: destination-key"));
            let first = headers.lines().next().unwrap();
            if first.starts_with("PUT /api/gallery/import/") {
                let descriptor = u32::from_be_bytes(body[..4].try_into().unwrap()) as usize;
                let output_length = u64::from_be_bytes(body[4..12].try_into().unwrap()) as usize;
                assert_eq!(output_length, output.len());
                let import: serde_json::Value =
                    serde_json::from_slice(&body[12..12 + descriptor]).unwrap();
                assert_eq!(import["metadata"], expected_metadata);
                assert_eq!(&body[12 + descriptor..], output.as_slice());
                return br#"{"filename":"copied-2.png"}"#.to_vec();
            }
            assert!(first.contains("/copied%2D2%2Epng/transfer "));
            if first.starts_with("GET ") {
                return destination_offer_bytes.clone();
            }
            let length = u32::from_be_bytes(body[..4].try_into().unwrap()) as usize;
            let descriptor: serde_json::Value =
                serde_json::from_slice(&body[4..4 + length]).unwrap();
            assert_eq!(descriptor["archive_identity_sha256"], destination_identity);
            assert!(descriptor["members"][0].get("member_id").is_none());
            assert_eq!(&body[4 + length..], private.as_slice());
            serde_json::to_vec(&serde_json::json!({"archive_identity_sha256": destination_identity, "member_count": 1})).unwrap()
        });
        let local = LocalServerInfo {
            kind: "external",
            base_url: destination.base_url,
            api_key: Some("destination-key".into()),
            port: 0,
        };
        let copied = mirror_to_local(source, "original.png".into(), Some(metadata), None, local)
            .await
            .unwrap();
        assert_eq!(copied, "copied-2.png");
        source_thread.join().unwrap();
        destination_thread.join().unwrap();
    }
}

async fn bounded_response(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, String> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err("Retained source-media response exceeds its limit.".into());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "Retained source-media transfer failed.")?
    {
        if bytes.len().saturating_add(chunk.len()) > limit {
            return Err("Retained source-media response exceeds its limit.".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}
fn request(
    client: &reqwest::Client,
    source: &MediaSaveTarget,
    path: &str,
) -> reqwest::RequestBuilder {
    let request = client.get(format!("{}{path}", source.base_url.trim_end_matches('/')));
    match source.api_key.as_deref().filter(|key| !key.is_empty()) {
        Some(key) => request.header("X-Api-Key", key),
        None => request,
    }
}
async fn offer(
    client: &reqwest::Client,
    source: &MediaSaveTarget,
    path: &str,
    allow_legacy: bool,
) -> Result<Option<Offer>, String> {
    let response = request(client, source, &format!("{path}/transfer"))
        .send()
        .await
        .map_err(|_| "Couldn't reach the retained source-media host.")?;
    if allow_legacy && matches!(response.status().as_u16(), 404 | 405) {
        let inventory = request(client, source, path)
            .send()
            .await
            .map_err(|_| "Couldn't check whether this print has retained source media.")?;
        if !inventory.status().is_success() {
            return Err("This host cannot safely copy the print's retained source media.".into());
        }
        let bytes = bounded_response(inventory, MAX_DESCRIPTOR_BYTES).await?;
        let inventory: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|_| "Invalid retained source-media inventory.")?;
        if inventory["availability"] == "unavailable_legacy" {
            return Ok(None);
        }
        return Err(
            "This host cannot copy the print's retained source media. Update it and try again."
                .into(),
        );
    }
    if !response.status().is_success() {
        return Err(format!(
            "Couldn't read retained source media (HTTP {}).",
            response.status().as_u16()
        ));
    }
    let bytes = bounded_response(response, MAX_DESCRIPTOR_BYTES).await?;
    let offer: Offer =
        serde_json::from_slice(&bytes).map_err(|_| "Invalid retained source-media offer.")?;
    validate(&offer)?;
    Ok(Some(offer))
}

/// Mirror through both servers' authoritative galleries. Private retained
/// bytes never enter the webview and no output-only filesystem fallback exists.
#[tauri::command]
pub async fn mirror_gallery_print(
    state: tauri::State<'_, AppState>,
    source: MediaSaveTarget,
    filename: String,
    metadata: Option<Box<mold_core::OutputMetadata>>,
    timestamp: Option<u64>,
) -> Result<String, String> {
    if !valid_filename(&filename) {
        return Err("Invalid gallery filename.".into());
    }
    let LocalGalleryAuthority::Server(local) = local_gallery_authority(&state).await else {
        return Err(
            "Start the local server before copying a print with its retained source media.".into(),
        );
    };
    mirror_to_local(source, filename, metadata, timestamp, local).await
}

async fn mirror_to_local(
    source: MediaSaveTarget,
    filename: String,
    metadata: Option<Box<mold_core::OutputMetadata>>,
    timestamp: Option<u64>,
    local: LocalServerInfo,
) -> Result<String, String> {
    let _permit = MIRROR_PERMITS
        .acquire()
        .await
        .map_err(|_| "The gallery media service is unavailable.")?;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(300))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| "Couldn't prepare the gallery transfer.")?;
    let encoded =
        percent_encoding::utf8_percent_encode(&filename, percent_encoding::NON_ALPHANUMERIC);
    let path = format!("/api/gallery/source-media/{encoded}");
    let origin = offer(&client, &source, &path, true).await?;
    if let Some(origin) = &origin {
        same_output(origin, origin)?;
        if !origin.output_sha256.as_deref().is_some_and(valid_digest)
            || !origin
                .output_size_bytes
                .is_some_and(|size| size > 0 && size <= MAX_GALLERY_MEDIA_BYTES as u64)
        {
            return Err("Invalid output identity or output size exceeds the copy limit.".into());
        }
        if origin.members.is_empty() && origin.metadata.as_deref().is_some_and(has_source_markers) {
            return Err(
                "This print's retained source media is unavailable; the copy would be incomplete."
                    .into(),
            );
        }
        if metadata.is_some()
            && serde_json::to_value(&metadata).ok() != serde_json::to_value(&origin.metadata).ok()
        {
            return Err(
                "The print's settings changed. Refresh the Library before copying it.".into(),
            );
        }
    }
    if origin.is_none() && metadata.as_deref().is_some_and(has_source_markers) {
        return Err(
            "This print's retained source media cannot be copied from this older host.".into(),
        );
    }
    let output = fetch_gallery_bytes(
        &client,
        &source,
        &format!("/api/gallery/image/{encoded}"),
        MAX_GALLERY_MEDIA_BYTES,
        "file",
        None,
    )
    .await?
    .bytes;
    if let Some(origin) = &origin {
        if origin.output_size_bytes != Some(output.len() as u64)
            || origin.output_sha256.as_deref() != Some(digest(&output).as_str())
        {
            return Err("The print changed while downloading. Try again.".into());
        }
        let current = offer(&client, &source, &path, false)
            .await?
            .ok_or("Retained source-media offer disappeared.")?;
        if origin.archive_identity_sha256 != current.archive_identity_sha256 {
            return Err("The print's retained source media changed. Try again.".into());
        }
        same_output(origin, &current)?;
    }
    let imported_metadata = origin
        .as_ref()
        .map(|origin| origin.metadata.clone())
        .unwrap_or(metadata);
    let destination_filename = save_output_bytes_server(
        local.clone(),
        filename,
        output,
        imported_metadata,
        timestamp,
    )
    .await?;
    let Some(origin) = origin else {
        return Ok(destination_filename);
    };
    let destination = MediaSaveTarget {
        base_url: local.base_url.clone(),
        api_key: Some(api_key(&local)?.to_string()),
    };
    let encoded_destination = percent_encoding::utf8_percent_encode(
        &destination_filename,
        percent_encoding::NON_ALPHANUMERIC,
    );
    let destination_path = format!("/api/gallery/source-media/{encoded_destination}");
    let destination_offer = offer(&client, &destination, &destination_path, false)
        .await?
        .ok_or("The local gallery cannot retain source media. Update it before copying.")?;
    same_output(&origin, &destination_offer)?;
    if origin.members.is_empty() {
        return Ok(destination_filename);
    }

    let mut members = origin.members.clone();
    for member in &mut members {
        member.member_id = None;
    }
    #[derive(Serialize)]
    struct Descriptor<'a> {
        archive_identity_sha256: &'a str,
        members: &'a [Member],
    }
    let descriptor = serde_json::to_vec(&Descriptor {
        archive_identity_sha256: &destination_offer.archive_identity_sha256,
        members: &members,
    })
    .map_err(|_| "Couldn't encode retained source-media transfer.")?;
    if descriptor.len() > MAX_DESCRIPTOR_BYTES {
        return Err("Retained source-media descriptor is too large.".into());
    }
    let mut prefix = Vec::with_capacity(4 + descriptor.len());
    prefix.extend_from_slice(&(descriptor.len() as u32).to_be_bytes());
    prefix.extend_from_slice(&descriptor);
    // Unnamed private staging is owned by the body stream and disappears on
    // success, transport failure, or cancellation. No server path crosses IPC.
    let staging =
        tempfile::tempfile().map_err(|_| "Couldn't stage retained source media privately.")?;
    let mut staging = tokio::fs::File::from_std(staging);
    staging
        .write_all(&prefix)
        .await
        .map_err(|_| "Couldn't stage retained source media privately.")?;
    for member in &origin.members {
        let member_id = member
            .member_id
            .as_deref()
            .ok_or("Retained source-media member has no identity.")?;
        let encoded_member =
            percent_encoding::utf8_percent_encode(member_id, percent_encoding::NON_ALPHANUMERIC);
        let response = request(&client, &source, &format!("{path}/{encoded_member}"))
            .send()
            .await
            .map_err(|_| "Couldn't download retained source media.")?;
        if !response.status().is_success() {
            return Err("The retained source media is unavailable; the copy is incomplete.".into());
        }
        let bytes = bounded_response(response, member.size_bytes as usize).await?;
        if bytes.len() as u64 != member.size_bytes || digest(&bytes) != member.sha256 {
            return Err("Retained source media changed during copying. Try again.".into());
        }
        staging
            .write_all(&bytes)
            .await
            .map_err(|_| "Couldn't stage retained source media privately.")?;
    }
    staging
        .seek(SeekFrom::Start(0))
        .await
        .map_err(|_| "Couldn't read private retained source-media staging.")?;
    let stream = futures_util::stream::try_unfold(staging, |mut file| async move {
        let mut chunk = vec![0; 64 * 1024];
        let read = file.read(&mut chunk).await?;
        if read == 0 {
            return Ok::<_, std::io::Error>(None);
        }
        chunk.truncate(read);
        Ok(Some((chunk, file)))
    });
    let response = client
        .put(format!(
            "{}{destination_path}/transfer",
            destination.base_url.trim_end_matches('/')
        ))
        .header(
            "X-Api-Key",
            destination
                .api_key
                .as_deref()
                .ok_or("The local gallery has no API key.")?,
        )
        .header(
            reqwest::header::CONTENT_TYPE,
            "application/vnd.mold.retained-media-transfer",
        )
        .body(reqwest::Body::wrap_stream(stream))
        .send()
        .await
        .map_err(|_| "Couldn't attach retained source media to the local print.")?;
    if !response.status().is_success() {
        return Err("The local print changed or its retained source media could not be attached. The copy is incomplete.".into());
    }
    #[derive(Deserialize)]
    struct Ack {
        archive_identity_sha256: String,
        member_count: usize,
    }
    let acknowledgement: Ack =
        serde_json::from_slice(&bounded_response(response, MAX_DESCRIPTOR_BYTES).await?)
            .map_err(|_| "Invalid retained source-media transfer acknowledgement.")?;
    if acknowledgement.archive_identity_sha256 != destination_offer.archive_identity_sha256
        || acknowledgement.member_count != origin.members.len()
    {
        return Err(
            "The retained source-media transfer was not confirmed; the copy is incomplete.".into(),
        );
    }
    Ok(destination_filename)
}
