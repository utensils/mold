//! `mold library` — browse and organize one serving host's live gallery.
//!
//! Non-grid commands always use `$MOLD_HOST` (`MOLD_API_KEY` when set) and
//! never fall back to direct filesystem access. `grid` delegates to the
//! existing TUI Library so terminal protocol detection and the merged gallery
//! stay singular.

use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};

use anyhow::{bail, Context, Result};
use colored::Colorize;
use mold_core::{
    Collection, CollectionCreateRequest, CollectionItemsRequest, CollectionUpdateRequest,
    GalleryBulkMutationRequest, GalleryImage, GalleryOrganizeRequest, GalleryPatchRequest,
    MoldClient, OutputFormat, ServerCapabilities,
};

use crate::{LibraryAction, LibraryCollectionAction, LibraryTagAction};

pub async fn run(action: LibraryAction) -> Result<()> {
    match action {
        LibraryAction::Grid { host, local } => library_grid(host, local).await,
        action => run_remote(action, &MoldClient::from_env()).await,
    }
}

async fn run_remote(action: LibraryAction, client: &MoldClient) -> Result<()> {
    match action {
        LibraryAction::List {
            query,
            tags,
            collection,
            favorite,
            format,
            limit,
            offset,
            json,
        } => {
            library_list(
                client,
                ListOptions {
                    query,
                    tags,
                    collection,
                    favorite,
                    format,
                    limit,
                    offset,
                    json,
                },
            )
            .await
        }
        LibraryAction::Show {
            filename,
            json,
            preview,
        } => library_show(client, &filename, json, preview).await,
        LibraryAction::Title {
            filename,
            title,
            clear,
        } => library_title(client, &filename, title, clear).await,
        LibraryAction::Favorite { filenames } => {
            mutate_prints(client, &filenames, Some(true), Vec::new(), Vec::new()).await
        }
        LibraryAction::Unfavorite { filenames } => {
            mutate_prints(client, &filenames, Some(false), Vec::new(), Vec::new()).await
        }
        LibraryAction::Tag { action } => library_tag(client, action).await,
        LibraryAction::Collection { action } => library_collection(client, action).await,
        LibraryAction::Trash { filenames } => library_trash(client, &filenames).await,
        LibraryAction::Export {
            filename,
            format,
            output,
        } => library_export(client, &filename, format, output.as_deref()).await,
        LibraryAction::Grid { .. } => unreachable!("grid handled before creating a client"),
    }
}

#[derive(Debug)]
struct ListOptions {
    query: Option<String>,
    tags: Vec<String>,
    collection: Option<String>,
    favorite: bool,
    format: Option<OutputFormat>,
    limit: usize,
    offset: usize,
    json: bool,
}

async fn library_list(client: &MoldClient, options: ListOptions) -> Result<()> {
    let collections = if options.collection.is_some() {
        require_organize(client).await?;
        client.list_collections().await?
    } else {
        client.list_collections().await.unwrap_or_default()
    };
    let collection_id = options
        .collection
        .as_deref()
        .map(|value| resolve_collection(&collections, value).map(|c| c.id.clone()))
        .transpose()?;
    let mut rows = client
        .list_gallery()
        .await
        .with_context(|| format!("could not list the Library on {}", client.host()))?;
    filter_and_sort(
        &mut rows,
        options.query.as_deref(),
        &options.tags,
        collection_id.as_deref(),
        options.favorite,
        options.format,
    );
    let total = rows.len();
    let page = rows
        .into_iter()
        .skip(options.offset)
        .take(options.limit)
        .collect::<Vec<_>>();

    if options.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "total": total,
                "offset": options.offset,
                "limit": options.limit,
                "items": page,
            }))?
        );
    } else {
        print!(
            "{}",
            render_listing(&page, total, options.offset, &collections)
        );
    }
    Ok(())
}

async fn library_show(
    client: &MoldClient,
    filename: &str,
    json: bool,
    preview: bool,
) -> Result<()> {
    let image = client
        .gallery_item(filename)
        .await
        .with_context(|| format!("could not read {filename} on {}", client.host()))?
        .ok_or_else(|| anyhow::anyhow!("Library print not found: {filename}"))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&image)?);
        return Ok(());
    }
    print!("{}", render_detail(&image));
    if preview {
        let bytes = preview_bytes(client, &image).await?;
        super::generate::preview_image(&bytes);
    }
    Ok(())
}

async fn preview_bytes(client: &MoldClient, image: &GalleryImage) -> Result<Vec<u8>> {
    let format = gallery_format(image);
    if format.is_some_and(|value| value.is_video()) {
        if let Some(bytes) = client.get_gallery_preview(&image.filename).await? {
            return Ok(bytes);
        }
        return client.get_gallery_thumbnail(&image.filename).await;
    }
    if format.is_some_and(|value| value.is_audio()) {
        return client.get_gallery_thumbnail(&image.filename).await;
    }
    client.get_gallery_image(&image.filename).await
}

async fn library_title(
    client: &MoldClient,
    filename: &str,
    title: Option<String>,
    clear: bool,
) -> Result<()> {
    require_organize(client).await?;
    let title = if clear { String::new() } else { title.unwrap() };
    let row = client
        .patch_gallery_image(
            filename,
            &GalleryPatchRequest {
                title: Some(title),
                ..Default::default()
            },
        )
        .await?;
    println!("{} {}", "updated".green(), display_title(&row));
    Ok(())
}

async fn library_tag(client: &MoldClient, action: LibraryTagAction) -> Result<()> {
    match action {
        LibraryTagAction::List { json } => {
            require_organize(client).await?;
            let tags = client.list_tags().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&tags)?);
            } else if tags.is_empty() {
                println!("No tags.");
            } else {
                for tag in tags {
                    println!("{:<64} {:>6}", tag.name, tag.count);
                }
            }
            Ok(())
        }
        LibraryTagAction::Add { filenames, tags } => {
            mutate_prints(client, &filenames, None, tags, Vec::new()).await
        }
        LibraryTagAction::Remove { filenames, tags } => {
            mutate_prints(client, &filenames, None, Vec::new(), tags).await
        }
        LibraryTagAction::Rename { old, new } => {
            require_organize(client).await?;
            client.rename_tag(&old, &new).await?;
            println!("{} {} → {}", "renamed".green(), old, new);
            Ok(())
        }
        LibraryTagAction::Delete { tag, yes } => {
            require_organize(client).await?;
            if !yes && !confirm(&format!("Delete tag \"{tag}\" from every print?"))? {
                bail!("tag deletion aborted");
            }
            client.delete_tag(&tag).await?;
            println!("{} {}", "deleted tag".green(), tag);
            Ok(())
        }
    }
}

async fn mutate_prints(
    client: &MoldClient,
    filenames: &[String],
    favorite: Option<bool>,
    add_tags: Vec<String>,
    remove_tags: Vec<String>,
) -> Result<()> {
    let capabilities = require_organize(client).await?;
    if capabilities.gallery.bulk_mutations {
        client
            .mutate_gallery_bulk(&GalleryBulkMutationRequest {
                operation_id: uuid::Uuid::new_v4().to_string(),
                filenames: filenames.to_vec(),
                favorite,
                add_tags,
                remove_tags,
                ..Default::default()
            })
            .await?;
    } else {
        client
            .organize_gallery(&GalleryOrganizeRequest {
                filenames: filenames.to_vec(),
                favorite,
                add_tags: (!add_tags.is_empty()).then_some(add_tags),
                remove_tags: (!remove_tags.is_empty()).then_some(remove_tags),
                ..Default::default()
            })
            .await?;
    }
    println!("{} {} print(s)", "updated".green(), filenames.len());
    Ok(())
}

async fn library_collection(client: &MoldClient, action: LibraryCollectionAction) -> Result<()> {
    require_organize(client).await?;
    match action {
        LibraryCollectionAction::List { json } => {
            let rows = client.list_collections().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
            } else if rows.is_empty() {
                println!("No collections.");
            } else {
                print!("{}", render_collections(&rows));
            }
            Ok(())
        }
        LibraryCollectionAction::Show { collection, json } => {
            let rows = client.list_collections().await?;
            let selected = resolve_collection(&rows, &collection)?;
            let detail = client.get_collection(&selected.id).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&detail)?);
            } else {
                println!(
                    "{} ({})",
                    detail.collection.name.bold(),
                    detail.collection.slug
                );
                if let Some(description) = detail.collection.description.as_deref() {
                    println!("{description}");
                }
                println!("{} print(s)", detail.filenames.len());
                for filename in detail.filenames {
                    println!("  {filename}");
                }
            }
            Ok(())
        }
        LibraryCollectionAction::Create { name, description } => {
            let row = client
                .create_collection(&CollectionCreateRequest { name, description })
                .await?;
            println!("{} {}", "created".green(), row.name);
            Ok(())
        }
        LibraryCollectionAction::Update {
            collection,
            name,
            description,
            clear_description,
            cover,
            clear_cover,
            hidden,
            visible,
        } => {
            if name.is_none()
                && description.is_none()
                && !clear_description
                && cover.is_none()
                && !clear_cover
                && !hidden
                && !visible
            {
                bail!("no collection changes given");
            }
            let rows = client.list_collections().await?;
            let selected = resolve_collection(&rows, &collection)?;
            let row = client
                .update_collection(
                    &selected.id,
                    &CollectionUpdateRequest {
                        name,
                        description: if clear_description {
                            Some(String::new())
                        } else {
                            description
                        },
                        cover_filename: if clear_cover {
                            Some(String::new())
                        } else {
                            cover
                        },
                        hidden: if hidden {
                            Some(true)
                        } else if visible {
                            Some(false)
                        } else {
                            None
                        },
                    },
                )
                .await?;
            println!("{} {}", "updated".green(), row.name);
            Ok(())
        }
        LibraryCollectionAction::Delete { collection, yes } => {
            let rows = client.list_collections().await?;
            let selected = resolve_collection(&rows, &collection)?;
            if !yes
                && !confirm(&format!(
                    "Delete collection \"{}\"? Its prints will remain in the Library.",
                    selected.name
                ))?
            {
                bail!("collection deletion aborted");
            }
            client.delete_collection(&selected.id).await?;
            println!("{} {}", "deleted collection".green(), selected.name);
            Ok(())
        }
        LibraryCollectionAction::Add {
            collection,
            filenames,
        } => collection_membership(client, &collection, filenames, true).await,
        LibraryCollectionAction::Remove {
            collection,
            filenames,
        } => collection_membership(client, &collection, filenames, false).await,
    }
}

async fn collection_membership(
    client: &MoldClient,
    reference: &str,
    filenames: Vec<String>,
    add: bool,
) -> Result<()> {
    let rows = client.list_collections().await?;
    let selected = resolve_collection(&rows, reference)?;
    client
        .set_collection_items(
            &selected.id,
            &CollectionItemsRequest {
                add: if add { filenames.clone() } else { Vec::new() },
                remove: if add { Vec::new() } else { filenames.clone() },
            },
        )
        .await?;
    println!(
        "{} {} print(s) {} {}",
        if add { "added" } else { "removed" }.green(),
        filenames.len(),
        if add { "to" } else { "from" },
        selected.name
    );
    Ok(())
}

async fn library_trash(client: &MoldClient, filenames: &[String]) -> Result<()> {
    let capabilities = client
        .capabilities()
        .await
        .with_context(|| format!("could not read Library capabilities on {}", client.host()))?;
    let enabled = capabilities
        .gallery
        .trash
        .as_ref()
        .is_some_and(|trash| trash.enabled);
    if !enabled {
        bail!(
            "{} does not advertise recoverable Library trash; no files were deleted",
            client.host()
        );
    }
    client.trash_gallery_images(filenames).await.with_context(|| {
        "the host stopped while moving the requested prints; earlier filenames may already be in trash, so run `mold library list` and `mold trash list` before retrying"
    })?;
    for filename in filenames {
        println!("{} {}", "trashed".green(), filename);
    }
    Ok(())
}

/// `mold library export <file> --format obj|stl|ply` — transcode one stored
/// mesh and write the result locally.
///
/// HTTP to `$MOLD_HOST` with no local fallback, like every other non-grid
/// `mold library` command: the print lives on the serving host, and reading
/// its output directory directly would answer for the wrong machine.
///
/// The gallery file is never renamed or replaced. Exporting is a download.
async fn library_export(
    client: &MoldClient,
    filename: &str,
    format: mold_core::MeshExportFormat,
    output: Option<&str>,
) -> Result<()> {
    if !std::path::Path::new(filename)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("glb"))
    {
        bail!("only stored .glb prints can be exported; '{filename}' is not one");
    }
    let capabilities = client
        .capabilities()
        .await
        .with_context(|| format!("could not read Library capabilities on {}", client.host()))?;
    // Absence reads as NO, which is the right answer for every host built
    // before mesh delivery existed — it has neither the file nor the route.
    let advertised = capabilities
        .mesh
        .as_ref()
        .map(|mesh| mesh.export_formats.as_slice())
        .unwrap_or_default();
    if !advertised.contains(&format) {
        bail!(
            "{} does not export meshes as {format}{}",
            client.host(),
            if advertised.is_empty() {
                String::new()
            } else {
                format!(
                    " (this host offers {})",
                    advertised
                        .iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        );
    }

    let bytes = client
        .export_gallery_mesh(filename, format)
        .await
        .with_context(|| format!("could not export {filename} as {format}"))?;

    let destination = output.map(str::to_string).unwrap_or_else(|| {
        format!(
            "{}.{}",
            std::path::Path::new(filename)
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("mold-mesh"),
            format.extension()
        )
    });
    if destination == "-" {
        io::stdout()
            .write_all(&bytes)
            .context("could not write the exported mesh to stdout")?;
        io::stdout().flush().ok();
        return Ok(());
    }
    std::fs::write(&destination, &bytes)
        .with_context(|| format!("could not write {destination}"))?;
    // Same channel every other `mold library` mutation reports on, and it
    // stays out of the way when the bytes went to stdout.
    println!(
        "{} {destination} ({} bytes)",
        "exported".green(),
        bytes.len()
    );
    Ok(())
}

async fn require_organize(client: &MoldClient) -> Result<ServerCapabilities> {
    let capabilities = client
        .capabilities()
        .await
        .with_context(|| format!("could not read Library capabilities on {}", client.host()))?;
    if !capabilities.gallery.organize {
        bail!(
            "{} cannot organize Library prints; upgrade the host or enable its metadata database",
            client.host()
        );
    }
    Ok(capabilities)
}

fn resolve_collection<'a>(collections: &'a [Collection], value: &str) -> Result<&'a Collection> {
    if let Some(row) = collections.iter().find(|row| row.id == value) {
        return Ok(row);
    }
    if let Some(row) = collections.iter().find(|row| row.slug == value) {
        return Ok(row);
    }
    let matches = collections
        .iter()
        .filter(|row| row.name.eq_ignore_ascii_case(value))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [row] => Ok(*row),
        [] => {
            let available = collections
                .iter()
                .map(|row| row.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "collection not found: {value}{}",
                if available.is_empty() {
                    String::new()
                } else {
                    format!("; available: {available}")
                }
            )
        }
        _ => bail!("collection name is ambiguous: {value}"),
    }
}

fn filter_and_sort(
    rows: &mut Vec<GalleryImage>,
    query: Option<&str>,
    tags: &[String],
    collection_id: Option<&str>,
    favorite: bool,
    format: Option<OutputFormat>,
) {
    let query = query.map(str::to_lowercase);
    rows.retain(|row| {
        let query_matches = query.as_ref().is_none_or(|needle| {
            row.filename.to_lowercase().contains(needle)
                || display_title(row).to_lowercase().contains(needle)
                || row.metadata.prompt.to_lowercase().contains(needle)
                || row.metadata.model.to_lowercase().contains(needle)
                || row
                    .tags
                    .iter()
                    .any(|tag| tag.to_lowercase().contains(needle))
        });
        let tags_match = tags
            .iter()
            .all(|wanted| row.tags.iter().any(|tag| tag.eq_ignore_ascii_case(wanted)));
        let collection_matches =
            collection_id.is_none_or(|id| row.collections.iter().any(|candidate| candidate == id));
        query_matches
            && tags_match
            && collection_matches
            && (!favorite || row.favorite)
            && format.is_none_or(|wanted| gallery_format(row) == Some(wanted))
    });
    rows.sort_by(|a, b| {
        b.timestamp
            .cmp(&a.timestamp)
            .then_with(|| a.filename.cmp(&b.filename))
    });
}

fn gallery_format(row: &GalleryImage) -> Option<OutputFormat> {
    row.format.or(row.metadata.output_format).or_else(|| {
        row.filename
            .rsplit_once('.')
            .and_then(|(_, ext)| ext.parse().ok())
    })
}

fn display_title(row: &GalleryImage) -> &str {
    row.title
        .as_deref()
        .or(row.metadata.title.as_deref())
        .filter(|title| !title.trim().is_empty())
        .unwrap_or("—")
}

fn render_listing(
    rows: &[GalleryImage],
    total: usize,
    offset: usize,
    collections: &[Collection],
) -> String {
    use std::fmt::Write as _;
    if rows.is_empty() {
        return "No Library prints matched.\n".to_string();
    }
    let names = collections
        .iter()
        .map(|row| (row.id.as_str(), row.name.as_str()))
        .collect::<HashMap<_, _>>();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<38} {:<24} {:<7} {:<4} {:<24} {}",
        "FILENAME".bold(),
        "TITLE".bold(),
        "FORMAT".bold(),
        "FAV".bold(),
        "TAGS".bold(),
        "COLLECTIONS".bold()
    );
    for row in rows {
        let collection_names = row
            .collections
            .iter()
            .map(|id| names.get(id.as_str()).copied().unwrap_or(id.as_str()))
            .collect::<Vec<_>>()
            .join(",");
        let _ = writeln!(
            out,
            "{:<38} {:<24} {:<7} {:<4} {:<24} {}",
            truncate(&row.filename, 38),
            truncate(display_title(row), 24),
            gallery_format(row)
                .map(|format| format.to_string())
                .unwrap_or_else(|| "—".to_string()),
            if row.favorite { "♥" } else { "—" },
            truncate(&row.tags.join(","), 24),
            collection_names,
        );
    }
    let _ = writeln!(
        out,
        "Showing {}–{} of {total}",
        offset + 1,
        offset + rows.len()
    );
    out
}

fn render_detail(row: &GalleryImage) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "{}", display_title(row).bold());
    let _ = writeln!(out, "Filename: {}", row.filename);
    let _ = writeln!(out, "Model: {}", row.metadata.model);
    let _ = writeln!(out, "Prompt: {}", row.metadata.prompt);
    let _ = writeln!(out, "Seed: {}", row.metadata.seed);
    let _ = writeln!(out, "Size: {}×{}", row.metadata.width, row.metadata.height);
    let _ = writeln!(out, "Favorite: {}", if row.favorite { "yes" } else { "no" });
    let _ = writeln!(
        out,
        "Tags: {}",
        if row.tags.is_empty() {
            "—".to_string()
        } else {
            row.tags.join(", ")
        }
    );
    let _ = writeln!(
        out,
        "Collections: {}",
        if row.collections.is_empty() {
            "—".to_string()
        } else {
            row.collections.join(", ")
        }
    );
    out
}

fn render_collections(rows: &[Collection]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<32} {:<32} {:>7} {}",
        "NAME".bold(),
        "SLUG".bold(),
        "PRINTS".bold(),
        "VISIBILITY".bold()
    );
    for row in rows {
        let _ = writeln!(
            out,
            "{:<32} {:<32} {:>7} {}",
            truncate(&row.name, 32),
            row.slug,
            row.count,
            if row.hidden { "hidden" } else { "visible" }
        );
    }
    out
}

fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        value.to_string()
    } else {
        let mut out = value
            .chars()
            .take(max.saturating_sub(3))
            .collect::<String>();
        out.push_str("...");
        out
    }
}

fn confirm(prompt: &str) -> Result<bool> {
    if !io::stdin().is_terminal() {
        bail!("confirmation requires an interactive terminal; pass --yes to proceed");
    }
    eprint!("{prompt} [y/N] ");
    io::stderr().flush().ok();
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

#[cfg(feature = "tui")]
async fn library_grid(host: Option<String>, local: bool) -> Result<()> {
    mold_tui::run_tui_with_options(mold_tui::TuiLaunchOptions {
        host,
        local,
        api_key: std::env::var("MOLD_API_KEY").ok(),
        initial_workspace: mold_tui::TuiInitialWorkspace::Library,
        strict_host: true,
    })
    .await
}

#[cfg(not(feature = "tui"))]
async fn library_grid(_host: Option<String>, _local: bool) -> Result<()> {
    bail!("Library grid requires a build with `--features tui`")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(filename: &str, timestamp: u64, tags: &[&str], favorite: bool) -> GalleryImage {
        serde_json::from_value(serde_json::json!({
            "filename": filename,
            "metadata": {
                "prompt": "night owl",
                "model": "flux-dev:q4",
                "seed": 1,
                "steps": 20,
                "guidance": 3.5,
                "width": 1024,
                "height": 1024,
                "version": "test"
            },
            "timestamp": timestamp,
            "tags": tags,
            "favorite": favorite,
            "collections": []
        }))
        .unwrap()
    }

    #[test]
    fn filters_use_tag_and_semantics_and_stable_order() {
        let mut rows = vec![
            image("b.png", 7, &["owl", "night"], true),
            image("a.png", 7, &["owl", "night"], true),
            image("c.png", 9, &["owl"], true),
        ];
        filter_and_sort(
            &mut rows,
            Some("owl"),
            &["owl".into(), "NIGHT".into()],
            None,
            true,
            Some(OutputFormat::Png),
        );
        assert_eq!(
            rows.iter()
                .map(|row| row.filename.as_str())
                .collect::<Vec<_>>(),
            vec!["a.png", "b.png"]
        );
    }

    #[test]
    fn collection_resolution_prefers_id_then_slug_then_name() {
        let rows = vec![Collection {
            id: "id-1".into(),
            name: "Night Owls".into(),
            slug: "night-owls".into(),
            description: None,
            cover_filename: None,
            hidden: false,
            count: 2,
            created_at: 1,
            updated_at: 2,
        }];
        for value in ["id-1", "night-owls", "NIGHT OWLS"] {
            assert_eq!(resolve_collection(&rows, value).unwrap().id, "id-1");
        }
        assert!(resolve_collection(&rows, "missing")
            .unwrap_err()
            .to_string()
            .contains("available: Night Owls"));
    }

    #[test]
    fn listing_empty_page_does_not_claim_a_range() {
        assert_eq!(
            render_listing(&[], 4, 4, &[]),
            "No Library prints matched.\n"
        );
    }
}
