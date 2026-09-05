# Metal memory on macOS

Mold separates installed shared RAM from the memory budget available to Metal.
Device details in Studio (web, desktop and mobile), `mold gpu list --json`, and
the devices API report the **inference host's** effective Metal capacity,
allocation headroom, working-set recommendation and kernel-limit mode. Older
hosts omit this optional telemetry.

The effective capacity is the smallest of Metal's working-set recommendation,
a positive `iogpu.wired_limit_mb` setting, and installed RAM minus
`max(15% of RAM, 8 GiB)`. Allocation headroom also accounts for Mold's existing
Metal allocations and live available host RAM. Allocated bytes include cached
Metal buffers; they do not measure other applications' GPU allocations. The
legacy shared-RAM totals remain hardware/system telemetry.

A kernel value of **0 means automatic**, not zero usable memory. An unavailable
supported probe prevents further Metal admission until it recovers; it does not
silently substitute all installed RAM. If this optional kernel key is absent,
a valid Metal recommendation can still supply a budget. CPU and CUDA behavior
is unchanged.

## Inspect this machine

```bash
mold system metal-memory status
mold system metal-memory status --json
```

These commands always inspect the **local machine**, ignoring `MOLD_HOST`.
They run before Mold config/database initialization. The allocation and headroom
in this command belong to its own inspection process. To inspect a running
server's allocations, use `mold gpu list --json` against that host instead.

## Change the local machine-wide limit

```bash
sudo mold system metal-memory set 16384
sudo mold system metal-memory reset
```

The integer is in MiB (16384 MiB = 16 GiB). `set` accepts positive unsigned
32-bit integers and refuses values that consume Mold's host-memory floor;
use `reset` for automatic mode. The explicit administration command requires
root, writes only `iogpu.wired_limit_mb`, and verifies the kernel readback.
Mold never invokes sudo internally, stores passwords, or elevates its server.
There is no remote HTTP/MCP mutation endpoint for this setting.

This is a system-wide setting affecting other GPU applications. It does not
reserve memory for Mold or guarantee a workload will fit. Mold refreshes its
budget at admission, load and existing streaming boundaries; changing the limit
does not asynchronously evict buffers from an active GPU command. A decrease
is clamped immediately on subsequent samples, even if Metal's recommendation
is stale. After an increase or reset, restart an **idle** inference process if
its recommendation has not refreshed; Mold retains the lower observed budget.

## Apply again after reboot

```bash
sudo mold system metal-memory set 16384 --persist
sudo mold system metal-memory reset --persist
```

Persistence installs only Mold's fixed root-owned boot policy at
`/Library/LaunchDaemons/io.utensils.mold.metal-memory.plist`. Its one-shot job
runs `/usr/sbin/sysctl -w iogpu.wired_limit_mb=16384` at subsequent boots; it does
not run Mold as root or continuously overwrite settings. Reset with `--persist`
removes that owned policy and any loaded registration. Without `--persist`,
set/reset changes the live value and reports any boot policy left in place.

Mold refuses foreign contents, symlinks and untrusted permissions. Concurrent
Mold administration is serialized. A persistence failure attempts conditional
rollback and reports partial state if another administrator has changed it;
inspect `status` before retrying. Other tools can still change this system-wide
key independently.

The kernel key is feature-detected and not a promised macOS compatibility API.
Metal's [working-set recommendation](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
is a performance recommendation, distinct from MLX's
[process residency control](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html).
Boot persistence follows Apple's
[LaunchDaemon lifecycle](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html).
