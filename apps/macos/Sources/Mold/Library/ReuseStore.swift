import Foundation
import MoldClient

/// What a Use These Settings is holding: which print's retained source media
/// the next render may be hydrated from, and anything the person should be
/// told about it.
///
/// Its own store rather than more state on `GenerateController`, which is
/// already over the type-size budget -- and because this outlives the click:
/// the probe runs while the recipe is already back in the draft, and the
/// answer arrives after the pane has drawn.
///
/// The RULE the whole thing rests on (`CLAUDE.md`, "Durable gallery source
/// media"): every client ALWAYS asks, because `OutputMetadata` under-reports
/// what a host kept -- inline `source_video`, `audio_file` and `mask_image`
/// bytes leave no marker at all. The metadata decides only whether an
/// unavailable answer is worth a sentence.
@MainActor
@Observable
final class ReuseStore {
    private let hosts: HostStore

    /// The print the draft on screen came from, once a machine has answered
    /// for it. `nil` means nothing to hydrate from -- a fresh draft, a reuse
    /// still in flight, or an edit that invalidated one.
    private(set) var authority: Authority?
    /// What to say about a print whose media cannot be restored. Shown once,
    /// beside the prompt, and cleared by the next reuse or the next submit.
    var notice: String?
    /// Monotonic fence. A second Use These Settings while the first probe is
    /// in the air must win: the older answer is for a print nobody is looking
    /// at any more, and installing it would hydrate the new render from the
    /// old print's archive.
    private var version = 0

    init(hosts: HostStore) { self.hosts = hosts }

    /// One print's retained media, on the ONE machine that holds it.
    struct Authority: Equatable, Sendable {
        let filename: String
        /// The machine whose archive owns the bytes -- never the machine the
        /// next render is going to, which may be another one entirely.
        let origin: MoldHost.ID
        let members: [RetainedSourceMedia.Member]
    }

    /// Opens a reuse. Clears whatever the last one left and returns the fence
    /// the probe must still be current against.
    @discardableResult
    func begin() -> Int {
        version += 1
        authority = nil
        notice = nil
        return version
    }

    /// Whether the answer in hand is still about the print on screen.
    func isCurrent(_ fence: Int) -> Bool { fence == version }

    /// Asks EVERY known copy of the print, in order, and keeps the first
    /// machine that can actually hand the media over.
    ///
    /// The Library lists a print once per machine that holds it, and mirroring
    /// an output does not copy the producing host's private archive -- so a
    /// copy answering `unavailable_legacy` says nothing about the machine that
    /// made it. A concrete archive or auth failure is preferred over a
    /// mirror's blank, and one unreachable copy must not hide a reachable
    /// archive (port of `useReuseStillPrint.ts:45-84`).
    func probe(_ copies: [PrintID], fence: Int, disclosing metadata: OutputMetadata) async {
        var unavailable: (PrintID, RetainedSourceMedia.Availability)?
        for copy in copies {
            guard isCurrent(fence) else { return }
            guard let client = hosts.backend(for: copy.host) else { continue }
            guard let inventory = try? await client.retainedSourceMedia(for: copy.filename)
            else { continue }
            guard isCurrent(fence) else { return }
            if inventory.availability == .available {
                authority = Authority(filename: copy.filename, origin: copy.host,
                                      members: inventory.members)
                return
            }
            if unavailable == nil || unavailable?.1 == .unavailableLegacy {
                unavailable = (copy, inventory.availability)
            }
        }
        guard isCurrent(fence), let unavailable else { return }
        // Only a print whose OWN metadata says conditioning bytes shipped is
        // worth a sentence. A text-to-image print's archive entry has no pins
        // either, and the host cannot tell the two apart.
        guard RetainedSourceMedia.disclosable(metadata) else { return }
        // Never over the top of one already said: the reuse may have had
        // something more immediate to report -- a model this machine no
        // longer has -- and one line is one line.
        guard notice == nil else { return }
        notice = RetainedSourceMedia.disclosure(unavailable.1)
    }

    /// A long clip is rendered as a CHAIN JOB, and `POST /api/chain-jobs` is
    /// not one of the three doors that redeem a reuse session -- only
    /// `/api/generate`, `/api/generate/stream` and `/api/generation-batches`
    /// do (`routes.rs:3080`, `:3475`, `:4662`). So a clip past the
    /// checkpoint's own clip size cannot be hydrated from the print's
    /// archive, and this SAYS so rather than rendering it without the picture
    /// it was supposed to start from.
    func warnIfTheRouteCannotCarryMedia(chained: Bool) {
        guard chained, authority != nil else { return }
        notice = "This clip is long enough that the machine renders it in "
            + "pieces, and that route can\u{2019}t restore the print\u{2019}s own source "
            + "picture. Attach one before developing."
    }

    /// Forgets the print entirely -- a new draft is not that print any more.
    func clear() {
        version += 1
        authority = nil
        notice = nil
    }
}

// A HANDLE IS NEVER HELD ACROSS AN EDIT, because one is never held at all.
// The host binds it to the sha256 of `target_request`, so the mint happens
// INSIDE the submit, against the exact request going out, and the handle is
// consumed by the very next call. There is no window in which the draft can
// move under a minted session, and nothing persists one -- `BatchAdmission`
// excludes it from its coding keys.
//
// An edit to a hydrated ROLE is handled structurally rather than by watching
// for it: `RetainedSourceMedia.members(_:forHydrating:)` reads the outgoing
// request at mint time and asks only for the roles it has no bytes for.
// Someone who reattached their own picture keeps it, the retained one is not
// requested, and the host's `RETAINED_MEDIA_REUSE_TARGET_CONFLICT` can never
// fire for something this client chose to send.
