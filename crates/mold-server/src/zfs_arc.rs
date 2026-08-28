//! Evictable ZFS ARC as host-RAM headroom (#1439).
//!
//! Linux's `MemAvailable` never counts the ZFS ARC, even though the kernel
//! shrinker reclaims it inline under allocation pressure (openzfs/zfs#10255,
//! open since 2020). Verifying a ~37 GB H3 artifact set through the page
//! cache therefore removed ~16 GB from the very sample that admits the
//! render, and mold refused work the kernel would have made room for.
//!
//! The credit is OpenZFS's OWN reclaimable figure — what `arc_evictable_memory`
//! (`module/os/linux/zfs/arc_os.c`) hands the shrinker and what
//! `arc_reduce_target_size` honours: `min(Σ mru/mfu evictable, max(size -
//! c_min, 0))`. Never `size` (pinned, dirty, and anonymous buffers are not
//! evictable), never `Σ evictable` alone (`c_min` is a hard floor), never
//! `size - arc_sys_free` (a free-RAM watermark, the wrong axis).
//!
//! It is ZERO whenever ZFS is not in a position to drain that much inline:
//! while `memory_available_bytes` is negative (ZFS is already self-evicting),
//! when `zfs_arc_pc_percent` is non-zero (the floor rises by a file-page term
//! this reader does not model), and when `zfs_arc_shrinker_limit` is non-zero
//! or unreadable (OpenZFS 2.2's default of 10000 pages caps each reclaim
//! pass at ~160 MiB, so a burst allocation can still OOM before the ARC
//! drains; 2.3 defaults it to 0 and TrueNAS forces 0).
//!
//! The reader enters the sample at exactly one place —
//! `resources::ram_snapshot()` — through `RamSnapshot::with_zfs_arc_credit`.
//! Absent arcstats is `None`, never an error; garbage is `None` plus one
//! DEBUG line for the life of the process.

#[cfg(target_os = "linux")]
pub(crate) const ARCSTATS_PATH: &str = "/proc/spl/kstat/zfs/arcstats";
#[cfg(target_os = "linux")]
pub(crate) const PC_PERCENT_PATH: &str = "/sys/module/zfs/parameters/zfs_arc_pc_percent";
#[cfg(target_os = "linux")]
pub(crate) const SHRINKER_LIMIT_PATH: &str = "/sys/module/zfs/parameters/zfs_arc_shrinker_limit";
/// `0` / `false` / `no` / `off` stops counting evictable ARC; unset or any
/// other value keeps the credit on.
pub(crate) const DISABLE_ENV: &str = "MOLD_HOST_RAM_ZFS_ARC";

/// The arcstats fields the credit is a function of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ArcStats {
    pub size: u64,
    pub c_min: u64,
    pub mru_evictable_data: u64,
    pub mru_evictable_metadata: u64,
    pub mfu_evictable_data: u64,
    pub mfu_evictable_metadata: u64,
    /// Signed (kstat type 3): negative means ZFS is under self-imposed
    /// pressure and already evicting. Optional because older modules may
    /// not publish it.
    pub memory_available_bytes: Option<i64>,
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ArcStatsParseError {
    #[error("arcstats missing `{0}`")]
    MissingField(&'static str),
    #[error("arcstats `{field}` is not a number: {value}")]
    BadNumber { field: &'static str, value: String },
}

/// ZFS module parameters that change what the shrinker will actually drain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ArcTunables {
    /// `zfs_arc_pc_percent`; non-zero raises the eviction floor by a
    /// file-page term this reader does not model, so the credit is zeroed.
    pub pc_percent: u64,
    /// `zfs_arc_shrinker_limit`; `None` when unreadable. Anything but a
    /// readable `0` zeroes the credit — a capped shrinker cannot drain a
    /// burst allocation's worth of ARC inline.
    pub shrinker_limit: Option<u64>,
}

impl ArcTunables {
    #[cfg(test)]
    pub(crate) const INLINE_RECLAIM: Self = Self {
        pc_percent: 0,
        shrinker_limit: Some(0),
    };
}

const REQUIRED_FIELDS: [&str; 6] = [
    "size",
    "c_min",
    "mru_evictable_data",
    "mru_evictable_metadata",
    "mfu_evictable_data",
    "mfu_evictable_metadata",
];

/// Parse `/proc/spl/kstat/zfs/arcstats`: two header lines, then
/// `name type data` rows in any order, with unknown rows ignored.
pub(crate) fn parse_arcstats(text: &str) -> Result<ArcStats, ArcStatsParseError> {
    let mut unsigned: [Option<u64>; 6] = [None; 6];
    let mut memory_available_bytes: Option<i64> = None;
    for line in text.lines().skip(2) {
        let mut columns = line.split_whitespace();
        let (Some(name), Some(_kind), Some(value)) =
            (columns.next(), columns.next(), columns.next())
        else {
            continue;
        };
        if let Some(index) = REQUIRED_FIELDS.iter().position(|field| *field == name) {
            let parsed = value
                .parse::<u64>()
                .map_err(|_| ArcStatsParseError::BadNumber {
                    field: REQUIRED_FIELDS[index],
                    value: value.to_string(),
                })?;
            unsigned[index] = Some(parsed);
        } else if name == "memory_available_bytes" {
            let parsed = value
                .parse::<i64>()
                .map_err(|_| ArcStatsParseError::BadNumber {
                    field: "memory_available_bytes",
                    value: value.to_string(),
                })?;
            memory_available_bytes = Some(parsed);
        }
    }
    let mut required = [0u64; 6];
    for (index, slot) in unsigned.iter().enumerate() {
        required[index] = slot.ok_or(ArcStatsParseError::MissingField(REQUIRED_FIELDS[index]))?;
    }
    let [size, c_min, mru_evictable_data, mru_evictable_metadata, mfu_evictable_data, mfu_evictable_metadata] =
        required;
    Ok(ArcStats {
        size,
        c_min,
        mru_evictable_data,
        mru_evictable_metadata,
        mfu_evictable_data,
        mfu_evictable_metadata,
        memory_available_bytes,
    })
}

impl ArcStats {
    /// OpenZFS's `arc_evictable_memory()` at `zfs_arc_pc_percent = 0`:
    /// `min(clean, max(size - c_min, 0))`, and nothing while the ARC is
    /// already under its own pressure.
    pub(crate) fn evictable_bytes(&self) -> u64 {
        if self.memory_available_bytes.is_some_and(|bytes| bytes < 0) {
            return 0;
        }
        let clean = self
            .mru_evictable_data
            .saturating_add(self.mru_evictable_metadata)
            .saturating_add(self.mfu_evictable_data)
            .saturating_add(self.mfu_evictable_metadata);
        clean.min(self.size.saturating_sub(self.c_min))
    }
}

/// The credit admission may spend, given the module's own tunables.
pub(crate) fn credit_from(stats: &ArcStats, tunables: ArcTunables) -> u64 {
    if tunables.pc_percent != 0 || tunables.shrinker_limit != Some(0) {
        return 0;
    }
    stats.evictable_bytes()
}

/// `MOLD_HOST_RAM_ZFS_ARC`: unset or any value but `0`/`false`/`no`/`off`
/// (case-insensitive) keeps the credit on — the same set `MOLD_MDNS` reads.
pub(crate) fn credit_enabled(env_value: Option<&str>) -> bool {
    match env_value {
        Some(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        None => true,
    }
}

/// The evictable ZFS ARC credit for this host, or `None` when there is no
/// ZFS (no arcstats), the credit is switched off, or the file is unreadable
/// or malformed. Read once per sample; a failure is logged once per process.
#[cfg(target_os = "linux")]
pub(crate) fn evictable_arc_credit() -> Option<u64> {
    if !credit_enabled(std::env::var(DISABLE_ENV).ok().as_deref()) {
        return None;
    }
    let text = match std::fs::read_to_string(ARCSTATS_PATH) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            log_once(format_args!(
                "{ARCSTATS_PATH} is unreadable; not counting evictable ZFS ARC: {error}"
            ));
            return None;
        }
    };
    let stats = match parse_arcstats(&text) {
        Ok(stats) => stats,
        Err(error) => {
            log_once(format_args!(
                "{ARCSTATS_PATH} did not parse; not counting evictable ZFS ARC: {error}"
            ));
            return None;
        }
    };
    let tunables = ArcTunables {
        pc_percent: read_u64_param(PC_PERCENT_PATH).unwrap_or(0),
        shrinker_limit: read_u64_param(SHRINKER_LIMIT_PATH),
    };
    Some(credit_from(&stats, tunables))
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn evictable_arc_credit() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn read_u64_param(path: &str) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

#[cfg(target_os = "linux")]
fn log_once(message: std::fmt::Arguments<'_>) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::debug!("{message}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// hal9000, OpenZFS 2.3.7-1, 2026-08-27 — the fact sheet's sample, laid
    /// out as the kernel prints it (two header lines, `name type data`, u64
    /// rows as type 4 and the signed `memory_available_bytes` as type 3),
    /// with rows this reader ignores interleaved.
    const HAL9000_2_3_7: &str = "\
24 1 0x01 147 39984 580965800 143899331450515
name                            type data
hits                            4    7773740345
misses                          4    14625879
size                            4    4305033792
compressed_size                 4    3237768192
uncompressed_size               4    4488095744
c                               4    4680966144
c_min                           4    2098436480
c_max                           4    17179869184
mru_size                        4    676482560
mru_evictable_data              4    533930496
mru_evictable_metadata          4    2890240
mfu_size                        4    3050733568
mfu_evictable_data              4    2471586304
mfu_evictable_metadata          4    21536256
anon_evictable_data             4    0
anon_evictable_metadata         4    0
arc_no_grow                     4    0
arc_need_free                   4    0
arc_sys_free                    4    2347199360
memory_all_bytes                4    67149967360
memory_free_bytes               4    41014341632
memory_available_bytes          3    38667142272
";

    fn hal9000() -> ArcStats {
        parse_arcstats(HAL9000_2_3_7).expect("the live sample parses")
    }

    #[test]
    fn parses_the_hal9000_2_3_7_sample_exactly() {
        let stats = hal9000();
        assert_eq!(
            stats,
            ArcStats {
                size: 4_305_033_792,
                c_min: 2_098_436_480,
                mru_evictable_data: 533_930_496,
                mru_evictable_metadata: 2_890_240,
                mfu_evictable_data: 2_471_586_304,
                mfu_evictable_metadata: 21_536_256,
                memory_available_bytes: Some(38_667_142_272),
            }
        );
        // clean = 3,029,943,296; size - c_min = 2,206,597,312; the floor wins.
        assert_eq!(stats.evictable_bytes(), 2_206_597_312);
        assert_eq!(
            credit_from(&stats, ArcTunables::INLINE_RECLAIM),
            2_206_597_312
        );
    }

    /// OpenZFS 2.2 has no `pd`/`pm`/`meta` triple and may order rows
    /// differently; the six fields this reader needs have been stable since
    /// 0.7 and `memory_available_bytes` is optional.
    #[test]
    fn a_2_2_shaped_file_parses_without_the_meta_triple() {
        let text = "\
13 1 0x01 120 1 2 3
name                            type data
mfu_evictable_metadata          4    10
some_future_counter             4    99
mfu_evictable_data              4    20
c_min                           4    100
mru_evictable_metadata          4    30
size                            4    1000
mru_evictable_data              4    40
l2_hits                         4    0
";
        let stats = parse_arcstats(text).unwrap();
        assert_eq!(
            stats,
            ArcStats {
                size: 1000,
                c_min: 100,
                mru_evictable_data: 40,
                mru_evictable_metadata: 30,
                mfu_evictable_data: 20,
                mfu_evictable_metadata: 10,
                memory_available_bytes: None,
            }
        );
        assert_eq!(stats.evictable_bytes(), 100);
    }

    #[test]
    fn truncated_or_garbage_text_is_refused_not_zeroed() {
        assert_eq!(
            parse_arcstats(""),
            Err(ArcStatsParseError::MissingField("size"))
        );
        let header_only = "24 1 0x01 147 39984 580965800 143899331450515\nname type data\n";
        assert_eq!(
            parse_arcstats(header_only),
            Err(ArcStatsParseError::MissingField("size"))
        );
        let cut = HAL9000_2_3_7
            .split_once("c_min")
            .map(|(head, _)| head)
            .unwrap();
        assert_eq!(
            parse_arcstats(cut),
            Err(ArcStatsParseError::MissingField("c_min"))
        );
        let garbage = HAL9000_2_3_7.replace(
            "size                            4    4305033792",
            "size                            4    abc",
        );
        assert_eq!(
            parse_arcstats(&garbage),
            Err(ArcStatsParseError::BadNumber {
                field: "size",
                value: "abc".to_string(),
            })
        );
        let signed_garbage = HAL9000_2_3_7.replace(
            "memory_available_bytes          3    38667142272",
            "memory_available_bytes          3    lots",
        );
        assert!(matches!(
            parse_arcstats(&signed_garbage),
            Err(ArcStatsParseError::BadNumber {
                field: "memory_available_bytes",
                ..
            })
        ));
    }

    #[test]
    fn memory_available_bytes_is_signed_and_negative_zeroes_the_credit() {
        let text = HAL9000_2_3_7.replace(
            "memory_available_bytes          3    38667142272",
            "memory_available_bytes          3    -1234567",
        );
        let stats = parse_arcstats(&text).unwrap();
        assert_eq!(stats.memory_available_bytes, Some(-1_234_567));
        assert_eq!(
            stats.evictable_bytes(),
            0,
            "ZFS already under its own pressure promises nothing more"
        );
        assert_eq!(credit_from(&stats, ArcTunables::INLINE_RECLAIM), 0);
    }

    #[test]
    fn the_credit_never_exceeds_size_minus_c_min_or_goes_negative() {
        let below_floor = ArcStats {
            size: 1_000,
            c_min: 2_000,
            mru_evictable_data: 900,
            mru_evictable_metadata: 0,
            mfu_evictable_data: 0,
            mfu_evictable_metadata: 0,
            memory_available_bytes: Some(1),
        };
        assert_eq!(below_floor.evictable_bytes(), 0);
        let more_clean_than_size = ArcStats {
            size: 5_000,
            c_min: 1_000,
            mru_evictable_data: 4_000,
            mru_evictable_metadata: 4_000,
            mfu_evictable_data: 4_000,
            mfu_evictable_metadata: 4_000,
            memory_available_bytes: None,
        };
        assert_eq!(more_clean_than_size.evictable_bytes(), 4_000);
        let overflowing = ArcStats {
            size: u64::MAX,
            c_min: 0,
            mru_evictable_data: u64::MAX,
            mru_evictable_metadata: u64::MAX,
            mfu_evictable_data: 1,
            mfu_evictable_metadata: 1,
            memory_available_bytes: None,
        };
        assert_eq!(overflowing.evictable_bytes(), u64::MAX);
    }

    #[test]
    fn a_nonzero_pc_percent_zeroes_the_credit() {
        let stats = hal9000();
        assert_eq!(
            credit_from(
                &stats,
                ArcTunables {
                    pc_percent: 50,
                    shrinker_limit: Some(0),
                }
            ),
            0
        );
    }

    /// OpenZFS 2.2 caps each shrinker pass at `zfs_arc_shrinker_limit` pages
    /// (default 10000 ≈ 160 MiB per kernel ask), so a burst allocation can
    /// OOM before the ARC drains. Only a readable `0` — 2.3's default and
    /// what TrueNAS forces — earns the credit.
    #[test]
    fn a_nonzero_or_unreadable_shrinker_limit_zeroes_the_credit() {
        let stats = hal9000();
        assert_eq!(
            credit_from(
                &stats,
                ArcTunables {
                    pc_percent: 0,
                    shrinker_limit: Some(10_000),
                }
            ),
            0
        );
        assert_eq!(
            credit_from(
                &stats,
                ArcTunables {
                    pc_percent: 0,
                    shrinker_limit: None,
                }
            ),
            0
        );
    }

    #[test]
    fn the_kill_switch_disables_the_credit() {
        assert!(credit_enabled(None));
        assert!(credit_enabled(Some("1")));
        assert!(credit_enabled(Some("on")));
        assert!(credit_enabled(Some("anything")));
        assert!(!credit_enabled(Some("0")));
        assert!(!credit_enabled(Some("off")));
        assert!(!credit_enabled(Some("false")));
        assert!(!credit_enabled(Some("no")));
        assert!(!credit_enabled(Some(" OFF ")));
    }

    /// The live reader never errors: on a ZFS host it answers the credit,
    /// elsewhere `None`, and whatever it answers respects the formula's cap.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_live_reader_answers_none_or_a_bounded_credit() {
        let credit = evictable_arc_credit();
        match std::fs::read_to_string(ARCSTATS_PATH) {
            Ok(text) => {
                let stats = parse_arcstats(&text).expect("a real arcstats parses");
                if let Some(credit) = credit {
                    assert!(
                        credit <= stats.size,
                        "credit {credit} exceeds the ARC size {}",
                        stats.size
                    );
                }
            }
            Err(_) => assert_eq!(credit, None),
        }
    }
}
