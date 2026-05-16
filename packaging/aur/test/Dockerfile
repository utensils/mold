# Headless Arch builder for local PKGBUILD smoke tests.
#
# Drive via scripts/aur/test-in-docker.sh, which mounts the repo at
# /workspace and runs `makepkg -si` against one of the in-tree
# PKGBUILDs, then sanity-checks the installed binary.
#
# Note for mold: the `-bin` PKGBUILD declares `depends=('cuda')`, which
# pulls in the full CUDA toolkit (~5 GB) on first install. Smoke tests
# install the package but `mold --version` works without a real GPU, so
# the build proves linkage without exercising CUDA at runtime.

# Pin linux/amd64 so the image runs under Rosetta on Apple Silicon and
# qemu-user elsewhere. Arch upstream only publishes amd64 to Docker Hub
# (Arch Linux ARM is a separate distro), so this is the only portable
# choice. On a native x86_64 host this is a no-op.
FROM --platform=linux/amd64 archlinux:latest

# pacman 6.1+ ships an internal seccomp sandbox that the qemu-user /
# Rosetta syscall ABI returns EINVAL for. Disable it up front — safe
# in a single-user throwaway container, but never in prod.
RUN printf '\nDisableSandbox\n' >> /etc/pacman.conf

# Bootstrap the keyring before any package install so the
# archlinux-keyring post-install hook doesn't whine about a missing
# trustdb on first sync.
RUN pacman-key --init && pacman-key --populate archlinux

# Refresh + install the build toolchain in one layer. `base-devel`
# pulls in makepkg, fakeroot, gcc, etc. `rust` + `cargo` cover the
# source-build PKGBUILDs. `git` is needed for VCS sources; `sudo` for
# makepkg's `pacman -U` step under the unprivileged builder user.
RUN pacman -Sy --noconfirm archlinux-keyring && \
    pacman -Syu --noconfirm \
        base-devel \
        git \
        sudo \
        rust \
        cargo \
        pkgconf && \
    rm -rf /var/cache/pacman/pkg/*

# makepkg refuses to run as root. Create a `builder` user with
# passwordless sudo so the entrypoint can install the built
# .pkg.tar.zst back into the system for the smoke test.
RUN useradd -m -G wheel -s /bin/bash builder && \
    echo "builder ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/00-builder && \
    chmod 0440 /etc/sudoers.d/00-builder

USER builder
WORKDIR /workspace

# /workspace is mounted by scripts/aur/test-in-docker.sh, pointing at
# the repo root. PKGBUILDs resolve to /workspace/packaging/aur/<pkg>/PKGBUILD.
VOLUME ["/workspace"]

CMD ["bash"]
