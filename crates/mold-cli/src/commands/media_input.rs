//! Resolve Library inputs against the selected authenticated host before authoring.
//! Private temporary files keep existing image/identity/probe validation paths intact.
use super::media_address::{self, MediaAddress};
use anyhow::{ensure, Context, Result};
use mold_core::{MoldClient, RetainedSourceMediaAvailability, RetainedSourceMediaMember};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

const MAX_INPUT_BYTES: usize = 256 * 1024 * 1024;

pub(crate) fn select_member<'a>(
    members: &'a [RetainedSourceMediaMember],
    id: &str,
) -> Result<&'a RetainedSourceMediaMember> {
    if let Some(exact) = members.iter().find(|member| member.member_id == id) {
        return Ok(exact);
    }
    ensure!((8..=64).contains(&id.len()) && id.bytes().all(|b| b.is_ascii_hexdigit()),
        "no retained member '{id}'; use a unique prefix of at least 8 hexadecimal characters (run `mold library source-media <filename>` for the ids)");
    let id = id.to_ascii_lowercase();
    let mut matches = members
        .iter()
        .filter(|member| member.member_id.to_ascii_lowercase().starts_with(&id));
    let found = matches
        .next()
        .with_context(|| format!("no retained member matches '{id}'"))?;
    ensure!(
        matches.next().is_none(),
        "ambiguous retained member prefix '{id}'; use more characters"
    );
    Ok(found)
}

#[derive(Clone, Copy)]
pub(crate) enum InputKind {
    Image,
    Video,
    Audio,
}

pub(crate) struct InputResolver {
    client: Option<MoldClient>,
    host: Option<String>,
    local: bool,
    directories: Vec<tempfile::TempDir>,
    pub(crate) remote_only: bool,
}

impl InputResolver {
    pub(crate) fn new(host: Option<&str>, local: bool) -> Self {
        Self {
            client: None,
            host: host.map(str::to_owned),
            local,
            directories: Vec::new(),
            remote_only: false,
        }
    }

    pub(crate) async fn resolve(&mut self, value: &str, kind: InputKind) -> Result<String> {
        let address = media_address::parse(value).map_err(anyhow::Error::msg).context("Library input addresses use print:FILENAME or source:FILENAME#MEMBER; prefix ./ to treat either prefix as a local filename")?;
        if matches!(address, MediaAddress::Local(_)) {
            return Ok(value.into());
        }
        ensure!(!self.local, "Library input addresses require the selected serving host; download a local file before using --local");
        self.remote_only = true;
        if self.client.is_none() {
            self.client = Some(crate::control::client_for_host(self.host.as_deref())?);
        }
        let client = self.client.as_ref().unwrap();
        let (bytes, name) = match address {
            MediaAddress::Print(filename) => (
                client.get_gallery_input(filename, MAX_INPUT_BYTES).await?,
                filename.to_owned(),
            ),
            MediaAddress::Source { filename, member } => {
                let inventory = client.gallery_source_media(filename).await?;
                ensure!(
                    inventory.availability == RetainedSourceMediaAvailability::Available,
                    "source media for '{filename}' is unavailable on {} ({:?})",
                    client.host(),
                    inventory.availability
                );
                let selected = select_member(&inventory.members, member)?;
                ensure!(
                    selected.size_bytes <= MAX_INPUT_BYTES as u64,
                    "source media exceeds the input limit"
                );
                let bytes = client
                    .download_gallery_source_media_content(filename, selected, MAX_INPUT_BYTES)
                    .await?;
                if let Some(expected) = selected.content_sha256.as_deref() {
                    ensure!(
                        format!("{:x}", Sha256::digest(&bytes)).eq_ignore_ascii_case(expected),
                        "source content digest does not match its inventory"
                    );
                }
                (bytes, selected.display_name.clone())
            }
            MediaAddress::Local(_) => unreachable!(),
        };
        match kind {
            InputKind::Image => {
                image::guess_format(&bytes).context(
                    "this input requires an image, not video/audio or a retained envelope",
                )?;
            }
            InputKind::Video => ensure!(
                bytes.get(4..8) == Some(&b"ftyp"[..])
                    || bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]),
                "this reference requires video content"
            ),
            InputKind::Audio => ensure!(
                bytes.starts_with(b"RIFF")
                    || bytes.starts_with(b"fLaC")
                    || bytes.starts_with(b"OggS")
                    || bytes.starts_with(b"ID3")
                    || bytes.get(4..8) == Some(&b"ftyp"[..])
                    || (bytes.len() >= 2 && bytes[0] == 0xff && bytes[1] & 0xf6 == 0xf0),
                "this reference requires audio content"
            ),
        }
        let filename = Path::new(&name)
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| {
                !name.is_empty()
                    && *name != "."
                    && *name != ".."
                    && !name.chars().any(char::is_control)
            })
            .unwrap_or("source-media");
        let dir = tempfile::Builder::new()
            .prefix("mold-library-input-")
            .tempdir()?;
        let path = dir.path().join(filename);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
        file.write_all(&bytes)?;
        self.directories.push(dir);
        Ok(path
            .to_str()
            .context("temporary input path is not UTF-8")?
            .to_owned())
    }

    pub(crate) async fn optional(
        &mut self,
        value: Option<String>,
        kind: InputKind,
    ) -> Result<Option<String>> {
        match value {
            Some(value) => Ok(Some(self.resolve(&value, kind).await?)),
            None => Ok(None),
        }
    }

    pub(crate) async fn optional_path(
        &mut self,
        value: Option<PathBuf>,
        kind: InputKind,
    ) -> Result<Option<PathBuf>> {
        match value {
            Some(value) => Ok(Some(PathBuf::from(
                self.resolve(value.to_str().context("input path is not UTF-8")?, kind)
                    .await?,
            ))),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn member(id: &str) -> RetainedSourceMediaMember {
        serde_json::from_value(serde_json::json!({"member_id":id,"role":"source_image","display_name":"image.png","size_bytes":10})).unwrap()
    }
    #[test]
    fn member_prefix_requires_unique_eight_hex_and_preserves_exact_legacy_ids() {
        let members = [
            member("abcdef0123456789"),
            member("abcdef0199999999"),
            member("legacy"),
        ];
        assert!(select_member(&members, "abcdef01")
            .unwrap_err()
            .to_string()
            .contains("ambiguous"));
        assert_eq!(
            select_member(&members, "ABCDEF012").unwrap().member_id,
            "abcdef0123456789"
        );
        assert!(select_member(&members, "abc").is_err());
        assert!(select_member(&members, "nope").is_err());
        assert_eq!(
            select_member(&members, "legacy").unwrap().member_id,
            "legacy"
        );
    }
    #[tokio::test]
    async fn local_mode_refuses_address_before_contacting_host() {
        let mut resolver = InputResolver::new(Some("http://127.0.0.1:1"), true);
        assert!(resolver
            .resolve("print:frame.png", InputKind::Image)
            .await
            .unwrap_err()
            .to_string()
            .contains("--local"));
        assert_eq!(
            resolver
                .resolve("./print:frame.png", InputKind::Image)
                .await
                .unwrap(),
            "./print:frame.png"
        );
    }
    #[tokio::test]
    async fn host_source_is_verified_materialized_and_cleaned_without_exposing_thumbnail_bytes() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };
        let server = MockServer::start().await;
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(2, 2)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        let bytes = bytes.into_inner();
        Mock::given(method("GET")).and(path("/api/gallery/source-media/clip.mp4"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "availability":"available", "members":[{"member_id":"abcdef0123456789", "role":"keyframe:0", "display_name":"frame.png", "size_bytes":bytes.len(), "content_sha256":format!("{:x}", Sha256::digest(&bytes))}]
            }))).mount(&server).await;
        Mock::given(method("GET"))
            .and(path(
                "/api/gallery/source-media/clip.mp4/abcdef0123456789/content",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
            .mount(&server)
            .await;
        let mut resolver = InputResolver::new(None, false);
        resolver.client = Some(MoldClient::new(&server.uri()));
        let file = resolver
            .resolve("source:clip.mp4#abcdef01", InputKind::Image)
            .await
            .unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), bytes);
        assert!(resolver.remote_only);
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
        drop(resolver);
        assert!(!Path::new(&file).exists());
    }
    #[tokio::test]
    async fn source_resolution_errors_never_become_filesystem_reads() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/gallery/source-media/a.png"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;
        let mut resolver = InputResolver::new(None, false);
        resolver.client = Some(MoldClient::new(&server.uri()));
        let error = resolver
            .resolve("source:a.png#abcdef01", InputKind::Image)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("unavailable"));
        assert!(resolver.directories.is_empty());
    }
}
