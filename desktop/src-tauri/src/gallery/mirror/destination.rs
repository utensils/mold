//! Destination transfers stream through private native files, never JS blobs.
use super::*;
use std::future::Future;
use std::path::Path;

static ACTIVE: LazyLock<Mutex<HashMap<String, Arc<CopyCancellation>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[derive(Default)]
struct CopyCancellation {
    cancelled: AtomicBool,
    notify: tokio::sync::Notify,
}
struct ActiveCopy {
    id: String,
    cancellation: Arc<CopyCancellation>,
}
impl ActiveCopy {
    fn new(id: String) -> Result<Self, String> {
        if uuid::Uuid::parse_str(&id).is_err() {
            return Err("Invalid Library copy operation.".into());
        }
        let mut active = ACTIVE
            .lock()
            .map_err(|_| "Library copy registry is unavailable.")?;
        if active.len() >= 128 && !active.contains_key(&id) {
            return Err("Too many Library copy operations.".into());
        }
        let cancellation = active.entry(id.clone()).or_default().clone();
        if Arc::strong_count(&cancellation) > 2 {
            return Err("Library copy operation is already active.".into());
        }
        Ok(Self { id, cancellation })
    }
}
impl Drop for ActiveCopy {
    fn drop(&mut self) {
        if let Ok(mut active) = ACTIVE.lock() {
            active.remove(&self.id);
        }
    }
}
async fn operation<T>(
    id: String,
    work: impl Future<Output = Result<T, String>>,
) -> Result<T, String> {
    let active = ActiveCopy::new(id)?;
    if active.cancellation.cancelled.load(Ordering::Acquire) {
        return Err("Library copy stopped.".into());
    }
    tokio::select! {
        biased;
        _ = active.cancellation.notify.notified() => Err("Library copy stopped.".into()),
        result = async {
            let _permit = MIRROR_PERMITS.acquire().await.map_err(|_| "Library copy service is unavailable.")?;
            work.await
        } => result,
    }
}
#[tauri::command]
pub fn library_copy_cancel(operation_id: String) -> Result<(), String> {
    if uuid::Uuid::parse_str(&operation_id).is_err() {
        return Err("Invalid Library copy operation.".into());
    }
    let mut active = ACTIVE
        .lock()
        .map_err(|_| "Library copy registry is unavailable.")?;
    if active.len() >= 128 && !active.contains_key(&operation_id) {
        return Err("Too many Library copy operations.".into());
    }
    let cancellation = active.entry(operation_id).or_default();
    cancellation.cancelled.store(true, Ordering::Release);
    cancellation.notify.notify_one();
    Ok(())
}
/// Called only after the JS invocation and any cancellation invocation settled.
#[tauri::command]
pub fn library_copy_release(operation_id: String) -> Result<(), String> {
    if uuid::Uuid::parse_str(&operation_id).is_err() {
        return Err("Invalid Library copy operation.".into());
    }
    let mut active = ACTIVE
        .lock()
        .map_err(|_| "Library copy registry is unavailable.")?;
    if active
        .get(&operation_id)
        .is_some_and(|token| Arc::strong_count(token) == 1)
    {
        active.remove(&operation_id);
    }
    Ok(())
}
#[derive(Clone, Deserialize, Serialize)]
pub struct NativeOffer {
    archive_identity_sha256: String,
    output_sha256: String,
    output_size_bytes: u64,
    members: Vec<Member>,
    #[serde(rename = "rawMetadata")]
    raw_metadata: String,
}
fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| "Could not prepare native Library transfer.".into())
}
fn encoded(filename: &str) -> Result<String, String> {
    if !valid_filename(filename) {
        return Err("Invalid gallery filename.".into());
    }
    Ok(
        percent_encoding::utf8_percent_encode(filename, percent_encoding::NON_ALPHANUMERIC)
            .to_string(),
    )
}
fn json_metadata(offer: &NativeOffer) -> Result<serde_json::Value, String> {
    if offer.raw_metadata.len() > MAX_DESCRIPTOR_BYTES {
        return Err("Print recipe exceeds its metadata limit.".into());
    }
    serde_json::from_str(&offer.raw_metadata).map_err(|_| "Invalid print recipe.".into())
}
fn validate_offer(offer: &NativeOffer) -> Result<(), String> {
    if !valid_digest(&offer.output_sha256)
        || offer.output_size_bytes == 0
        || offer.output_size_bytes > MAX_TRANSFER_BYTES
    {
        return Err("Invalid output identity or output exceeds native copy limits.".into());
    }
    let metadata = json_metadata(offer)?;
    let typed = Offer {
        archive_identity_sha256: offer.archive_identity_sha256.clone(),
        members: offer.members.clone(),
        output_sha256: Some(offer.output_sha256.clone()),
        output_size_bytes: Some(offer.output_size_bytes),
        metadata: Some(Box::new(
            serde_json::from_value(metadata).map_err(|_| "Invalid print recipe.")?,
        )),
    };
    validate(&typed)?;
    let total = offer
        .members
        .iter()
        .try_fold(offer.output_size_bytes, |total, member| {
            total.checked_add(member.size_bytes)
        })
        .ok_or("Native copy exceeds its aggregate limit.")?;
    if total > MAX_TRANSFER_BYTES {
        return Err("Native copy exceeds 512 MiB including retained inputs.".into());
    }
    Ok(())
}
fn same_source(expected: &NativeOffer, actual: &NativeOffer) -> Result<(), String> {
    validate_offer(expected)?;
    validate_offer(actual)?;
    if expected.archive_identity_sha256 != actual.archive_identity_sha256
        || expected.output_sha256 != actual.output_sha256
        || expected.output_size_bytes != actual.output_size_bytes
        || json_metadata(expected)? != json_metadata(actual)?
        || serde_json::to_value(&expected.members).ok()
            != serde_json::to_value(&actual.members).ok()
    {
        return Err("The print or retained source media changed before native transfer.".into());
    }
    Ok(())
}
async fn read_offer(
    client: &reqwest::Client,
    target: &MediaSaveTarget,
    filename: &str,
) -> Result<Option<NativeOffer>, String> {
    let name = encoded(filename)?;
    let response = request(
        client,
        target,
        &format!("/api/gallery/source-media/{name}/transfer"),
    )
    .secure_send()
    .await
    .map_err(|_| "Could not read native retained-media offer.")?;
    if matches!(response.status().as_u16(), 404 | 405) {
        return Ok(None);
    }
    if !response.status().is_success() {
        return Err(format!(
            "Could not read retained source media (HTTP {}).",
            response.status().as_u16()
        ));
    }
    let value: serde_json::Value =
        serde_json::from_slice(&bounded_response(response, 1024 * 1024).await?)
            .map_err(|_| "Invalid retained-media offer.")?;
    let offer = NativeOffer {
        archive_identity_sha256: value["archive_identity_sha256"]
            .as_str()
            .ok_or("Missing archive identity.")?
            .to_owned(),
        output_sha256: value["output_sha256"]
            .as_str()
            .ok_or("Missing output digest.")?
            .to_owned(),
        output_size_bytes: value["output_size_bytes"]
            .as_u64()
            .ok_or("Missing output size.")?,
        members: serde_json::from_value(value["members"].clone())
            .map_err(|_| "Invalid retained members.")?,
        raw_metadata: serde_json::to_string(&value["metadata"])
            .map_err(|_| "Invalid print recipe.")?,
    };
    validate_offer(&offer)?;
    Ok(Some(offer))
}
#[derive(PartialEq)]
struct SourceSnapshot {
    metadata: serde_json::Value,
    timestamp: u64,
    metadata_synthetic: bool,
}
async fn source_snapshot(
    client: &reqwest::Client,
    target: &MediaSaveTarget,
    filename: &str,
) -> Result<SourceSnapshot, String> {
    let response = request(
        client,
        target,
        &format!("/api/gallery?filename={}", encoded(filename)?),
    )
    .secure_send()
    .await
    .map_err(|_| "Could not inspect exact source print provenance.")?;
    if !response.status().is_success() {
        return Err("Exact source print inventory is unavailable.".into());
    }
    let rows: Vec<serde_json::Value> =
        serde_json::from_slice(&bounded_response(response, 1024 * 1024).await?)
            .map_err(|_| "Invalid source print inventory.")?;
    let row = rows
        .into_iter()
        .find(|row| row["filename"] == filename)
        .ok_or("The source print is no longer in the live Library.")?;
    let metadata = row
        .get("metadata")
        .filter(|metadata| metadata.is_object())
        .cloned()
        .ok_or("The source print has no authenticated recipe.")?;
    let timestamp = row["timestamp"]
        .as_u64()
        .filter(|value| *value <= 9_007_199_254_740_991)
        .ok_or("The source print has an invalid creation timestamp.")?;
    let metadata_synthetic = match row.get("metadata_synthetic") {
        None | Some(serde_json::Value::Null) => false,
        Some(value) => value
            .as_bool()
            .ok_or("Invalid source print provenance marker.")?,
    };
    Ok(SourceSnapshot {
        metadata,
        timestamp,
        metadata_synthetic,
    })
}
fn import_descriptor(
    snapshot: &SourceSnapshot,
    archive: serde_json::Value,
    embedded: Option<serde_json::Value>,
) -> Result<Vec<u8>, String> {
    let synthetic = snapshot.metadata_synthetic && embedded.is_none();
    serde_json::to_vec(&serde_json::json!({"metadata":archive,"metadata_synthetic":synthetic,"timestamp":snapshot.timestamp})).map_err(|_| "Could not encode native import recipe.".into())
}
async fn legacy_metadata(
    client: &reqwest::Client,
    target: &MediaSaveTarget,
    filename: &str,
) -> Result<serde_json::Value, String> {
    let name = encoded(filename)?;
    let response = request(client, target, &format!("/api/gallery/source-media/{name}"))
        .secure_send()
        .await
        .map_err(|_| "Could not verify legacy source media.")?;
    if !response.status().is_success() {
        return Err("Legacy source media could not be verified.".into());
    }
    let inventory: serde_json::Value =
        serde_json::from_slice(&bounded_response(response, MAX_DESCRIPTOR_BYTES).await?)
            .map_err(|_| "Invalid legacy input inventory.")?;
    if inventory["availability"] != "unavailable_legacy"
        || inventory["members"]
            .as_array()
            .is_none_or(|members| !members.is_empty())
    {
        return Err(
            "Original inputs are damaged, unauthorized, or unverified; copy refused.".into(),
        );
    }
    let response = request(client, target, &format!("/api/gallery?filename={name}"))
        .secure_send()
        .await
        .map_err(|_| "Could not inspect legacy print.")?;
    if !response.status().is_success() {
        return Err("Legacy print inventory is unavailable.".into());
    }
    let rows: Vec<serde_json::Value> =
        serde_json::from_slice(&bounded_response(response, 1024 * 1024).await?)
            .map_err(|_| "Invalid legacy print inventory.")?;
    let metadata = rows
        .into_iter()
        .find(|row| row["filename"] == filename)
        .and_then(|row| row.get("metadata").cloned())
        .ok_or("Legacy print is not in the live Library.")?;
    let typed: mold_core::OutputMetadata =
        serde_json::from_value(metadata.clone()).map_err(|_| "Invalid legacy print recipe.")?;
    if has_source_markers(&typed) {
        return Err("This older source-bearing print cannot verify its originals. Update the source machine.".into());
    }
    Ok(metadata)
}
struct OutputFile {
    file: tempfile::NamedTempFile,
    digest: String,
    length: u64,
}
async fn download(
    client: &reqwest::Client,
    target: &MediaSaveTarget,
    path: &str,
    limit: u64,
) -> Result<OutputFile, String> {
    let response = request(client, target, path)
        .secure_send()
        .await
        .map_err(|_| "Could not download original Library bytes.")?;
    if !response.status().is_success() {
        return Err(format!(
            "Original bytes unavailable (HTTP {}).",
            response.status().as_u16()
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length > limit)
    {
        return Err("Native Library transfer exceeds its declared byte limit.".into());
    }
    let file =
        tempfile::NamedTempFile::new().map_err(|_| "Could not stage original bytes privately.")?;
    let mut output = tokio::fs::File::from_std(
        file.reopen()
            .map_err(|_| "Could not open private staging.")?,
    );
    let mut stream = response.bytes_stream();
    let mut digest = Sha256::new();
    let mut length = 0_u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| "Library transfer was interrupted.")?;
        length = length
            .checked_add(chunk.len() as u64)
            .ok_or("Transfer byte count overflow.")?;
        if length > limit {
            return Err("Native Library transfer exceeds its declared byte limit.".into());
        }
        digest.update(&chunk);
        output
            .write_all(&chunk)
            .await
            .map_err(|_| "Could not stage original bytes privately.")?;
    }
    output
        .flush()
        .await
        .map_err(|_| "Could not finish private staging.")?;
    Ok(OutputFile {
        file,
        digest: digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        length,
    })
}
fn verified_file(file: &OutputFile, expected: &NativeOffer) -> Result<(), String> {
    if file.digest != expected.output_sha256 || file.length != expected.output_size_bytes {
        return Err("Original output changed during native transfer.".into());
    }
    Ok(())
}
fn embedded(path: &Path, filename: &str) -> Option<serde_json::Value> {
    let format = mold_db::metadata_io::format_from_path(Path::new(filename))?;
    mold_db::metadata_io::read_embedded(path, format)
        .and_then(|metadata| serde_json::to_value(metadata).ok())
}
fn short_version(value: &str) -> &str {
    let Some((version, decoration)) = value.rsplit_once(" (") else {
        return value;
    };
    let Some(decoration) = decoration.strip_suffix(')') else {
        return value;
    };
    let Some((revision, date)) = decoration.split_once(' ') else {
        return value;
    };
    if (7..=40).contains(&revision.len())
        && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        && date.len() == 10
        && date.bytes().enumerate().all(|(index, byte)| {
            if index == 4 || index == 7 {
                byte == b'-'
            } else {
                byte.is_ascii_digit()
            }
        })
    {
        version
    } else {
        value
    }
}
fn metadata_matches(
    mut left: serde_json::Value,
    mut right: serde_json::Value,
    embedded: Option<&serde_json::Value>,
) -> bool {
    let (Some(left), Some(right)) = (left.as_object_mut(), right.as_object_mut()) else {
        return false;
    };
    for key in ["job_id", "generation_time_ms", "has_alpha"] {
        if left.get(key).is_none_or(serde_json::Value::is_null)
            || right.get(key).is_none_or(serde_json::Value::is_null)
        {
            left.remove(key);
            right.remove(key);
        }
    }
    if let (Some(a), Some(b)) = (
        left.get("version").and_then(serde_json::Value::as_str),
        right.get("version").and_then(serde_json::Value::as_str),
    ) {
        if a != b
            && short_version(a) == short_version(b)
            && (a == short_version(a) || b == short_version(b))
        {
            let version = serde_json::Value::String(short_version(a).to_string());
            left.insert("version".into(), version.clone());
            right.insert("version".into(), version);
        }
    }
    if let Some(embedded) = embedded {
        for key in ["scheduler", "transparent_background"] {
            let a = left.get(key).filter(|value| !value.is_null());
            let b = right.get(key).filter(|value| !value.is_null());
            if a.is_none() != b.is_none()
                && embedded
                    .get(key)
                    .is_some_and(|recorded| !recorded.is_null() && Some(recorded) == a.or(b))
            {
                left.remove(key);
                right.remove(key);
            }
        }
    }
    left == right
}
fn put(client: &reqwest::Client, target: &MediaSaveTarget, path: &str) -> reqwest::RequestBuilder {
    let request = client
        .put(format!("{}{path}", target.base_url.trim_end_matches('/')))
        .header("x-mold-connection-mode", target.connection_mode.as_str());
    match target.api_key.as_deref().filter(|key| !key.is_empty()) {
        Some(key) => request.header("X-Api-Key", key),
        None => request,
    }
}
fn output_import_request(
    client: &reqwest::Client,
    target: &MediaSaveTarget,
    path: &str,
    instance: &str,
) -> reqwest::RequestBuilder {
    put(client, target, path)
        .header(reqwest::header::CONTENT_TYPE, GALLERY_IMPORT_CONTENT_TYPE)
        .header("x-mold-destination-instance", instance)
}
fn body(file: &tempfile::NamedTempFile, prefix: Vec<u8>) -> Result<reqwest::Body, String> {
    let file = tokio::fs::File::from_std(
        file.reopen()
            .map_err(|_| "Could not open private transfer staging.")?,
    );
    let stream =
        futures_util::stream::try_unfold((file, Some(prefix)), |(mut file, prefix)| async move {
            if let Some(prefix) = prefix {
                return Ok::<_, std::io::Error>(Some((prefix, (file, None))));
            }
            let mut chunk = vec![0; 64 * 1024];
            let length = file.read(&mut chunk).await?;
            if length == 0 {
                return Ok(None);
            }
            chunk.truncate(length);
            Ok(Some((chunk, (file, None))))
        });
    Ok(reqwest::Body::wrap_stream(stream))
}
async fn current_offer(
    client: &reqwest::Client,
    target: &MediaSaveTarget,
    filename: &str,
    expected: &NativeOffer,
) -> Result<(), String> {
    if let Some(actual) = read_offer(client, target, filename).await? {
        return same_source(expected, &actual);
    }
    let metadata = legacy_metadata(client, target, filename).await?;
    if !expected.members.is_empty() || metadata != json_metadata(expected)? {
        return Err("Legacy print changed during copying.".into());
    }
    Ok(())
}
async fn installation(
    client: &reqwest::Client,
    target: &MediaSaveTarget,
    instance: &str,
) -> Result<(), String> {
    if instance.is_empty() {
        return Err("Reconnect this machine before copying.".into());
    }
    let response = request(client, target, "/api/status")
        .secure_send()
        .await
        .map_err(|_| "Could not verify the machine installation.")?;
    if !response.status().is_success() {
        return Err("Could not verify the machine installation.".into());
    }
    let value: serde_json::Value =
        serde_json::from_slice(&bounded_response(response, 1024 * 1024).await?)
            .map_err(|_| "Invalid machine identity response.")?;
    if value["instance_id"].as_str() != Some(instance) {
        return Err("The machine installation changed. Reconnect before copying.".into());
    }
    Ok(())
}
#[tauri::command]
pub async fn library_copy_legacy_offer(
    operation_id: String,
    source: MediaSaveTarget,
    source_instance: String,
    filename: String,
) -> Result<NativeOffer, String> {
    operation(operation_id, async {
        let client = client()?;
        installation(&client, &source, &source_instance).await?;
        if read_offer(&client, &source, &filename).await?.is_some() {
            return Err("The source transfer contract changed. Refresh before copying.".into());
        }
        let metadata = legacy_metadata(&client, &source, &filename).await?;
        let output = download(
            &client,
            &source,
            &format!("/api/gallery/image/{}", encoded(&filename)?),
            MAX_TRANSFER_BYTES,
        )
        .await?;
        if legacy_metadata(&client, &source, &filename).await? != metadata {
            return Err("The legacy print changed while downloading.".into());
        }
        if let Some(embedded) = embedded(output.file.path(), &filename) {
            if !metadata_matches(metadata.clone(), embedded.clone(), Some(&embedded)) {
                return Err(
                    "The original output's embedded recipe conflicts with its archive.".into(),
                );
            }
        }
        installation(&client, &source, &source_instance).await?;
        let raw_metadata =
            serde_json::to_string(&metadata).map_err(|_| "Invalid legacy recipe.")?;
        let archive_identity_sha256 =
            digest(format!("{}{}", output.digest, raw_metadata).as_bytes());
        let offer = NativeOffer {
            archive_identity_sha256,
            output_sha256: output.digest,
            output_size_bytes: output.length,
            members: vec![],
            raw_metadata,
        };
        validate_offer(&offer)?;
        Ok(offer)
    })
    .await
}
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn library_copy_output(
    operation_id: String,
    source: MediaSaveTarget,
    destination: MediaSaveTarget,
    source_instance: String,
    destination_instance: String,
    filename: String,
    expected: NativeOffer,
) -> Result<String, String> {
    operation(operation_id, async {
        let client = client()?; validate_offer(&expected)?;
        installation(&client, &source, &source_instance).await?; installation(&client, &destination, &destination_instance).await?;
        current_offer(&client, &source, &filename, &expected).await?;
        let snapshot = source_snapshot(&client, &source, &filename).await?;
        if !expected.members.is_empty() {
            let response = request(&client, &destination, "/api/capabilities").secure_send().await.map_err(|_| "Could not verify destination retention support.")?;
            if !response.status().is_success() { return Err("The destination cannot confirm retained-media transfer support.".into()); }
            let caps: serde_json::Value = serde_json::from_slice(&bounded_response(response, 1024 * 1024).await?).map_err(|_| "Invalid destination retention capabilities.")?;
            if !destination_can_retain_sources(&caps) { return Err("The destination cannot retain this print's original source media. Update it before copying.".into()); }
        }
        let output = download(&client, &source, &format!("/api/gallery/image/{}", encoded(&filename)?), expected.output_size_bytes).await?;
        verified_file(&output, &expected)?;
        current_offer(&client, &source, &filename, &expected).await?;
        let archive = json_metadata(&expected)?;
        let embedded = embedded(output.file.path(), &filename);
        if embedded.as_ref().is_some_and(|recipe| !metadata_matches(archive.clone(), recipe.clone(), Some(recipe))) { return Err("Original output and archived recipe disagree.".into()); }
        if !metadata_matches(snapshot.metadata.clone(), archive.clone(), embedded.as_ref()) { return Err("The source listing recipe changed before copying.".into()); }
        if source_snapshot(&client, &source, &filename).await? != snapshot { return Err("The source print provenance changed during copying.".into()); }
        let descriptor = import_descriptor(&snapshot, archive, embedded)?;
        if descriptor.len() > MAX_DESCRIPTOR_BYTES { return Err("Native import recipe is too large.".into()); }
        let mut prefix = Vec::with_capacity(12 + descriptor.len());
        prefix.extend_from_slice(&(descriptor.len() as u32).to_be_bytes()); prefix.extend_from_slice(&output.length.to_be_bytes()); prefix.extend_from_slice(&descriptor);
        installation(&client, &destination, &destination_instance).await?;
        let response = output_import_request(&client, &destination, &format!("/api/gallery/import/{}", encoded(&filename)?), &destination_instance)
            .body(body(&output.file, prefix)?).secure_send().await.map_err(|_| "Native output import was interrupted; retry repairs the copy.")?;
        if !response.status().is_success() { return Err(response_error(response).await); }
        let result: GalleryImportResponse = serde_json::from_slice(&bounded_response(response, MAX_DESCRIPTOR_BYTES).await?).map_err(|_| "Invalid native output import acknowledgement.")?;
        encoded(&result.filename)?;
        installation(&client, &destination, &destination_instance).await?;
        Ok(result.filename)
    }).await
}
async fn append_member(
    client: &reqwest::Client,
    source: &MediaSaveTarget,
    filename: &str,
    member: &Member,
    staging: &mut tokio::fs::File,
) -> Result<(), String> {
    let member_id = member
        .member_id
        .as_deref()
        .ok_or("Source member has no identity.")?;
    let path = format!(
        "/api/gallery/source-media/{}/{}",
        encoded(filename)?,
        percent_encoding::utf8_percent_encode(member_id, percent_encoding::NON_ALPHANUMERIC)
    );
    let response = request(client, source, &path)
        .secure_send()
        .await
        .map_err(|_| "Could not download retained source media.")?;
    if !response.status().is_success()
        || response
            .content_length()
            .is_some_and(|length| length > member.size_bytes)
    {
        return Err("Retained source media is unavailable or changed.".into());
    }
    let mut stream = response.bytes_stream();
    let mut digest = Sha256::new();
    let mut length = 0_u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| "Retained source transfer was interrupted.")?;
        length = length
            .checked_add(chunk.len() as u64)
            .ok_or("Retained member length overflow.")?;
        if length > member.size_bytes {
            return Err("Retained source exceeds its captured size.".into());
        }
        digest.update(&chunk);
        staging
            .write_all(&chunk)
            .await
            .map_err(|_| "Could not stage retained sources privately.")?;
    }
    let digest: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if length != member.size_bytes || digest != member.sha256 {
        return Err("Retained source digest changed during copying.".into());
    }
    Ok(())
}
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn library_copy_sources(
    operation_id: String,
    source: MediaSaveTarget,
    destination: MediaSaveTarget,
    source_instance: String,
    destination_instance: String,
    source_filename: String,
    destination_filename: String,
    expected_source: NativeOffer,
    expected_destination: NativeOffer,
) -> Result<(), String> {
    operation(operation_id, async {
        let client = client()?; validate_offer(&expected_source)?; validate_offer(&expected_destination)?;
        installation(&client, &source, &source_instance).await?; installation(&client, &destination, &destination_instance).await?;
        current_offer(&client, &source, &source_filename, &expected_source).await?;
        let actual_destination = read_offer(&client, &destination, &destination_filename).await?.ok_or("Destination retained transfer is unsupported.")?;
        same_source(&expected_destination, &actual_destination)?;
        if expected_source.output_sha256 != expected_destination.output_sha256 || expected_source.output_size_bytes != expected_destination.output_size_bytes { return Err("Retained inputs cannot attach to different output bytes.".into()); }
        if !metadata_matches(json_metadata(&expected_source)?, json_metadata(&expected_destination)?, None) {
            let output = download(&client, &source, &format!("/api/gallery/image/{}", encoded(&source_filename)?), expected_source.output_size_bytes).await?;
            verified_file(&output, &expected_source)?;
            let recipe = embedded(output.file.path(), &source_filename);
            if !recipe.as_ref().is_some_and(|recipe| metadata_matches(json_metadata(&expected_source).unwrap_or_default(), json_metadata(&expected_destination).unwrap_or_default(), Some(recipe))) {
                return Err("Retained sources cannot attach to a different or uncorroborated recipe.".into());
            }
        }
        if expected_source.members.is_empty() { return Ok(()); }
        let mut members = expected_source.members.clone(); for member in &mut members { member.member_id = None; }
        let descriptor = serde_json::to_vec(&serde_json::json!({"archive_identity_sha256":expected_destination.archive_identity_sha256,"members":members})).map_err(|_| "Could not encode retained transfer descriptor.")?;
        if descriptor.len() > MAX_DESCRIPTOR_BYTES { return Err("Retained descriptor exceeds its limit.".into()); }
        let staging = tempfile::NamedTempFile::new().map_err(|_| "Could not stage retained media privately.")?;
        let mut file = tokio::fs::File::from_std(staging.reopen().map_err(|_| "Could not open private retained staging.")?);
        for member in &expected_source.members { append_member(&client, &source, &source_filename, member, &mut file).await?; }
        file.flush().await.map_err(|_| "Could not finish private retained staging.")?;
        current_offer(&client, &source, &source_filename, &expected_source).await?;
        installation(&client, &destination, &destination_instance).await?;
        let current = read_offer(&client, &destination, &destination_filename).await?.ok_or("Destination retained authority disappeared.")?;
        same_source(&expected_destination, &current)?;
        let mut prefix = Vec::with_capacity(4 + descriptor.len()); prefix.extend_from_slice(&(descriptor.len() as u32).to_be_bytes()); prefix.extend_from_slice(&descriptor);
        let response = put(&client, &destination, &format!("/api/gallery/source-media/{}/transfer", encoded(&destination_filename)?))
            .header(reqwest::header::CONTENT_TYPE, "application/vnd.mold.retained-media-transfer").body(body(&staging, prefix)?).secure_send().await.map_err(|_| "Retained transfer was interrupted; retry repairs the copy.")?;
        if !response.status().is_success() { return Err(response_error(response).await); }
        let acknowledgement: serde_json::Value = serde_json::from_slice(&bounded_response(response, MAX_DESCRIPTOR_BYTES).await?).map_err(|_| "Invalid retained transfer acknowledgement.")?;
        if acknowledgement["archive_identity_sha256"] != expected_destination.archive_identity_sha256 || acknowledgement["member_count"].as_u64() != Some(expected_source.members.len() as u64) { return Err("Retained source publication was not confirmed; retry repairs the copy.".into()); }
        installation(&client, &destination, &destination_instance).await?; Ok(())
    }).await
}
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn library_copy_matches(
    operation_id: String,
    source: MediaSaveTarget,
    destination: MediaSaveTarget,
    source_instance: String,
    destination_instance: String,
    source_filename: String,
    destination_filename: String,
    origin: NativeOffer,
    candidate: NativeOffer,
) -> Result<bool, String> {
    operation(operation_id, async {
        let client = client()?;
        validate_offer(&origin)?;
        validate_offer(&candidate)?;
        if origin.output_sha256 != candidate.output_sha256
            || origin.output_size_bytes != candidate.output_size_bytes
        {
            return Ok(false);
        }
        installation(&client, &source, &source_instance).await?;
        installation(&client, &destination, &destination_instance).await?;
        current_offer(&client, &source, &source_filename, &origin).await?;
        current_offer(&client, &destination, &destination_filename, &candidate).await?;
        let left = json_metadata(&origin)?;
        let right = json_metadata(&candidate)?;
        if metadata_matches(left.clone(), right.clone(), None) {
            return Ok(true);
        }
        let output = download(
            &client,
            &source,
            &format!("/api/gallery/image/{}", encoded(&source_filename)?),
            origin.output_size_bytes,
        )
        .await?;
        verified_file(&output, &origin)?;
        let recipe = embedded(output.file.path(), &source_filename);
        Ok(recipe
            .as_ref()
            .is_some_and(|recipe| metadata_matches(left, right, Some(recipe))))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_import_preserves_timestamp_and_synthetic_provenance_until_embedded_proof() {
        let metadata: serde_json::Value =
            serde_json::from_str(r#"{"seed":18446744073709551615}"#).unwrap();
        let snapshot = SourceSnapshot {
            metadata: metadata.clone(),
            timestamp: 123456,
            metadata_synthetic: true,
        };
        let descriptor: serde_json::Value =
            serde_json::from_slice(&import_descriptor(&snapshot, metadata.clone(), None).unwrap())
                .unwrap();
        assert_eq!(descriptor["timestamp"], 123456);
        assert_eq!(descriptor["metadata_synthetic"], true);
        assert_eq!(descriptor["metadata"]["seed"].as_u64(), Some(u64::MAX));
        let descriptor: serde_json::Value = serde_json::from_slice(
            &import_descriptor(&snapshot, metadata.clone(), Some(metadata)).unwrap(),
        )
        .unwrap();
        assert_eq!(descriptor["timestamp"], 123456);
        assert_eq!(descriptor["metadata_synthetic"], false);
    }
    #[test]
    fn native_import_keeps_full_archive_recipe_after_embedded_compatibility() {
        let archive: serde_json::Value = serde_json::from_str(r#"{"seed":18446744073709551615,"prompt":"literal\\n\r\nnext","negative_prompt":"literal\\t","job_id":"archive-only","version":"1.0"}"#).unwrap();
        let mut embedded = archive.clone();
        embedded.as_object_mut().unwrap().remove("job_id");
        let snapshot = SourceSnapshot {
            metadata: archive.clone(),
            timestamp: 123,
            metadata_synthetic: false,
        };
        assert!(metadata_matches(
            archive.clone(),
            embedded.clone(),
            Some(&embedded)
        ));
        let descriptor: serde_json::Value = serde_json::from_slice(
            &import_descriptor(&snapshot, archive.clone(), Some(embedded)).unwrap(),
        )
        .unwrap();
        assert_eq!(descriptor["metadata"], archive);
    }
    #[test]
    fn native_import_request_captures_destination_installation() {
        let target = MediaSaveTarget {
            connection_mode: Default::default(),
            base_url: "https://destination.invalid".into(),
            api_key: Some("fixture-private".into()),
        };
        let request = output_import_request(
            &reqwest::Client::new(),
            &target,
            "/api/gallery/import/print.mp4",
            "captured-instance",
        )
        .build()
        .unwrap();
        assert_eq!(
            request.headers()["x-mold-destination-instance"],
            "captured-instance"
        );
        assert_eq!(
            request.headers()[reqwest::header::CONTENT_TYPE],
            GALLERY_IMPORT_CONTENT_TYPE
        );
    }
    #[test]
    fn mirror_recipe_compatibility_keeps_exact_u64_and_conflicting_facts() {
        let left: serde_json::Value = serde_json::from_str(
            r#"{"seed":18446744073709551615,"version":"1.0.0","scheduler":"euler"}"#,
        )
        .unwrap();
        let right: serde_json::Value = serde_json::from_str(r#"{"seed":18446744073709551615,"version":"1.0.0 (abcdef0 2026-10-18)","scheduler":"euler","has_alpha":false}"#).unwrap();
        assert!(metadata_matches(left.clone(), right.clone(), None));
        let mut changed = right.clone();
        changed["seed"] = serde_json::json!(18446744073709551614_u64);
        assert!(!metadata_matches(left.clone(), changed, None));
        let mut missing = right.clone();
        missing.as_object_mut().unwrap().remove("scheduler");
        assert!(!metadata_matches(left.clone(), missing.clone(), None));
        assert!(metadata_matches(left, missing, Some(&right)));
        let mut alpha = right.clone();
        alpha["has_alpha"] = serde_json::json!(true);
        assert!(!metadata_matches(right, alpha, None));
    }
    #[tokio::test]
    async fn cancellation_before_registration_is_not_lost_and_registry_is_cleaned() {
        let id = uuid::Uuid::new_v4().to_string();
        library_copy_cancel(id.clone()).unwrap();
        let result = operation(id.clone(), async { Ok::<_, String>("must not run") }).await;
        assert!(result.is_err());
        assert!(!ACTIVE.lock().unwrap().contains_key(&id));
    }
    #[tokio::test]
    async fn stop_interrupts_pending_native_network_work() {
        let id = uuid::Uuid::new_v4().to_string();
        let running = id.clone();
        let task = tokio::spawn(async move {
            operation(running, std::future::pending::<Result<(), String>>()).await
        });
        for _ in 0..100 {
            if ACTIVE.lock().unwrap().contains_key(&id) {
                break;
            }
            tokio::task::yield_now().await;
        }
        library_copy_cancel(id.clone()).unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap()
            .is_err());
        assert!(!ACTIVE.lock().unwrap().contains_key(&id));
    }
    #[test]
    fn repeated_pre_registration_or_late_cancels_retire_their_tombstones() {
        for _ in 0..256 {
            let id = uuid::Uuid::new_v4().to_string();
            library_copy_cancel(id.clone()).unwrap();
            library_copy_release(id.clone()).unwrap();
            assert!(!ACTIVE.lock().unwrap().contains_key(&id));
        }
    }
    #[test]
    fn duplicate_native_operation_cannot_replace_its_cancellation_authority() {
        let id = uuid::Uuid::new_v4().to_string();
        let first = ActiveCopy::new(id.clone()).unwrap();
        assert!(ActiveCopy::new(id.clone()).is_err());
        assert!(ACTIVE.lock().unwrap().contains_key(&id));
        drop(first);
        assert!(!ACTIVE.lock().unwrap().contains_key(&id));
    }
}
