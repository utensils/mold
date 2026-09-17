import Foundation

/// What a machine says while you are watching it.
///
/// **The frame name is not the type.** `GET /api/events` opens with
/// `event: authority`, and everything after it arrives as the literal
/// `event: event` with the real tag in the payload's `type` field. Routing on
/// the SSE name finds "event" for every gallery change and decodes none of
/// them, which is a stream that looks connected and reports nothing.
public enum MoldEvent: Hashable, Sendable {
    /// The opening frame, naming the machine. It is the fleet identity, so
    /// cached per-host state can be fenced on it: the same address answering
    /// with a different `instance_id` is a different library.
    case authority(instanceID: String)
    /// The server's buffer overran and this client missed deltas. Repair from
    /// the listings; do not carry on.
    case resyncRequired
    case gallery(Gallery)
    /// A device's lifecycle preference or runtime state changed on this
    /// machine. Carries NOTHING: the frame names three of `DeviceInfo`'s
    /// fourteen fields and the pane draws eight, so a partial patch would show
    /// a switch that flipped beside an activity that did not. It is an
    /// invalidation -- go and read `/api/devices`.
    case deviceStateChanged
    /// One job's lifecycle moved. See `Job` -- `MoldEvent+Job.swift`.
    case job(Job)
    /// The queue itself changed shape, rather than one row in it. See
    /// `Queue` -- `MoldEvent+Job.swift`.
    case queue(Queue)

    public enum Gallery: Hashable, Sendable {
        /// `row` present means insert without asking again; absent means the
        /// metadata DB did not record it and the caller must go and read.
        case added(filename: String, row: GalleryPrint?)
        case updated(filename: String, row: GalleryPrint?)
        case restored(filename: String, row: GalleryPrint?)
        case removed(filename: String)
        case trashed(filename: String)
        case collectionsChanged
    }

    /// Decodes one frame, or nothing.
    ///
    /// Nothing is the normal answer: chain jobs and whatever mold adds next
    /// all come down this stream too, and a client that treats an
    /// unrecognised tag as a failure breaks on a server upgrade.
    public init?(name: String?, data: String) {
        guard let bytes = data.data(using: .utf8) else { return nil }
        if name == "authority" {
            guard let frame = try? MoldJSON.decoder.decode(Authority.self, from: bytes)
            else { return nil }
            self = .authority(instanceID: frame.instanceId)
            return
        }
        if name == "resync_required" {
            self = .resyncRequired
            return
        }
        guard let frame = try? MoldJSON.decoder.decode(Frame.self, from: bytes) else { return nil }
        switch frame.type {
        case "gallery_added": self = .gallery(.added(filename: frame.name, row: frame.image))
        case "gallery_updated": self = .gallery(.updated(filename: frame.name, row: frame.image))
        case "gallery_restored": self = .gallery(.restored(filename: frame.name, row: frame.image))
        case "gallery_removed": self = .gallery(.removed(filename: frame.name))
        case "gallery_trashed": self = .gallery(.trashed(filename: frame.name))
        case "gallery_collections_changed": self = .gallery(.collectionsChanged)
        case "device_state_changed": self = .deviceStateChanged
        case "job_queued": self = .job(.queued(id: frame.jobId, model: frame.modelName))
        case "job_started":
            self = .job(.started(id: frame.jobId, model: frame.modelName, gpu: frame.gpu))
        case "job_ended": self = .job(.ended(id: frame.jobId))
        case "job_state_committed": self = .job(.stateCommitted(id: frame.jobId))
        case "generation_states_committed": self = .job(.statesCommitted)
        case "queue_paused": self = .queue(.paused)
        case "queue_resumed": self = .queue(.resumed)
        case "queue_plan_changed": self = .queue(.planChanged)
        // `chain_job_queued`, `chain_job_started`, `chain_job_ended`: left
        // undecoded on purpose. Old clients ignoring an unknown `type` is
        // exactly why chain jobs never inherit print-queue affordances
        // (reorder, `DELETE /api/queue/:id`) they do not support
        // (`types.rs:13209-13214`); scripted sequences are out of this app.
        // Every other unrecognised tag is a future server this build has not
        // learned yet, and the correct answer is silence, not a crash.
        default: return nil
        }
    }

    private struct Authority: Decodable {
        // Spelled as the wire spells it: `MoldJSON.decoder` converts from
        // snake case, and `instance_id` arrives as `instanceId`.
        let instanceId: String
    }

    private struct Frame: Decodable {
        let type: String
        let filename: String?
        let image: GalleryPrint?
        /// `job_queued`/`job_started`/`job_ended`/`job_state_committed` all
        /// name the job as `id`; `Frame.id` would collide with `Identifiable`
        /// conventions elsewhere, so it is read out under its own name.
        let id: String?
        let model: String?
        let gpu: Int?

        /// Collections changing names no print, and the gallery verbs all do.
        var name: String { filename ?? "" }
        var jobId: String { id ?? "" }
        var modelName: String { model ?? "" }
    }
}
