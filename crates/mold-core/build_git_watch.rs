use std::path::{Path, PathBuf};

pub fn watch_paths(package: &Path) -> Vec<PathBuf> {
    fn git(package: &Path, args: &[&str]) -> Option<String> {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(package)
            .args(args)
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8(output.stdout).ok())
            .flatten()
            .map(|value| value.trim().to_owned())
    }
    let mut names = vec!["HEAD".to_owned(), "packed-refs".to_owned()];
    if let Some(reference) = git(package, &["symbolic-ref", "-q", "HEAD"]) {
        names.push(reference);
    }
    let mut paths = names
        .into_iter()
        .filter_map(|name| {
            git(
                package,
                &["rev-parse", "--path-format=absolute", "--git-path", &name],
            )
            .map(|value| {
                let mut path = PathBuf::from(value);
                // Packed branches acquire a new loose ref on the next commit;
                // HEAD and packed-refs themselves do not change in that case.
                if name.starts_with("refs/") {
                    while !path.exists() && path.pop() {}
                }
                path
            })
        })
        .filter(|path| path.exists())
        .collect::<Vec<_>>();
    paths.sort();
    paths.dedup();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    fn git(root: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }
    #[test]
    fn linked_worktree_watchers_exist_and_track_the_real_head() {
        let root = std::env::temp_dir().join(format!(
            "mold-git-watch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let repo = root.join("repo");
        std::fs::create_dir(&repo).unwrap();
        git(&repo, &["init", "--initial-branch=main"]);
        git(
            &repo,
            &[
                "-c",
                "user.name=Mold test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-m",
                "fixture",
            ],
        );
        let ordinary_package = repo.join("crates/mold-core");
        std::fs::create_dir_all(&ordinary_package).unwrap();
        let ordinary = watch_paths(&ordinary_package);
        let branch = PathBuf::from(git(
            &repo,
            &[
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "refs/heads/main",
            ],
        ));
        assert!(ordinary.contains(&branch));
        assert!(ordinary.iter().all(|path| path.exists()));
        git(&repo, &["pack-refs", "--all"]);
        assert!(!branch.exists());
        let packed = watch_paths(&ordinary_package);
        let before = packed
            .iter()
            .map(|path| std::fs::metadata(path).unwrap().modified().unwrap())
            .collect::<Vec<_>>();
        git(
            &repo,
            &[
                "-c",
                "user.name=Mold test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-m",
                "new loose ref",
            ],
        );
        assert!(branch.exists());
        let detects_new_loose_ref = packed
            .iter()
            .zip(before)
            .any(|(path, before)| std::fs::metadata(path).unwrap().modified().unwrap() != before);
        let linked = root.join("linked");
        git(
            &repo,
            &["worktree", "add", "--detach", linked.to_str().unwrap()],
        );
        let package = linked.join("crates/mold-core");
        std::fs::create_dir_all(&package).unwrap();
        assert!(linked.join(".git").is_file());
        let paths = watch_paths(&package);
        let expected = PathBuf::from(git(
            &package,
            &["rev-parse", "--path-format=absolute", "--git-path", "HEAD"],
        ));
        assert!(!paths.is_empty());
        let valid = paths.iter().all(|path| path.exists()) && paths.contains(&expected);
        git(
            &repo,
            &["worktree", "remove", "--force", linked.to_str().unwrap()],
        );
        std::fs::remove_dir_all(&root).unwrap();
        assert!(
            detects_new_loose_ref,
            "a commit after pack-refs must invalidate the embedded SHA"
        );
        assert!(
            valid,
            "missing watched paths force every Cargo invocation to rebuild"
        );
    }
    #[test]
    fn archive_without_git_has_no_permanently_missing_watchers() {
        assert!(watch_paths(&std::env::temp_dir()).is_empty());
    }
}
