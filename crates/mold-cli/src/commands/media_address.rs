//! Host-scoped Library input addresses. Prefix local paths with `./` to escape.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MediaAddress<'a> {
    Local(&'a str),
    Print(&'a str),
    Source { filename: &'a str, member: &'a str },
}

pub(crate) fn parse(value: &str) -> Result<MediaAddress<'_>, String> {
    fn filename(value: &str) -> Result<&str, String> {
        if value.is_empty()
            || value == "."
            || value == ".."
            || value.contains(['/', '\\'])
            || value.chars().any(char::is_control)
        {
            return Err("a Library address needs one exact gallery filename, not a path".into());
        }
        Ok(value)
    }
    if let Some(value) = value.strip_prefix("print:") {
        return Ok(MediaAddress::Print(filename(value)?));
    }
    if let Some(value) = value.strip_prefix("source:") {
        let (owner, member) = value
            .rsplit_once('#')
            .ok_or("a source address needs # followed by a member id")?;
        if !(8..=64).contains(&member.len()) || !member.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(
                "a source member needs a unique 8-64 hexadecimal character id prefix".into(),
            );
        }
        return Ok(MediaAddress::Source {
            filename: filename(owner)?,
            member,
        });
    }
    Ok(MediaAddress::Local(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordinary_paths_and_explicit_escape_stay_local() {
        for path in [
            "./source:frame.png#abcdef01",
            "/tmp/image.png",
            "-",
            "foo.png",
        ] {
            assert_eq!(parse(path), Ok(MediaAddress::Local(path)));
        }
    }
    #[test]
    fn addresses_keep_exact_filename_and_member_prefix() {
        assert_eq!(
            parse("print:one # two.png"),
            Ok(MediaAddress::Print("one # two.png"))
        );
        assert_eq!(
            parse("source:one # two.png#Abcd1234"),
            Ok(MediaAddress::Source {
                filename: "one # two.png",
                member: "Abcd1234"
            })
        );
    }
    #[test]
    fn malformed_addresses_never_become_local_paths() {
        for value in [
            "print:",
            "print:../a.png",
            "print:a/b.png",
            "source:a.png",
            "source:a.png#abcdefg1",
            "source:a.png#abcdef0",
            "source:a.png#",
            "source:..#abcdef01",
            "print:a\nb.png",
        ] {
            assert!(parse(value).is_err(), "{value}");
        }
    }
}
