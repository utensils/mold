- **The first `queued` progress event no longer counts the job itself.** A
  request submitted behind one running generation could be told it was `#2 in
line` and then re-announced as `#1`: the seed read the live-job count, and
  when the durable queue feeder registered the new job before the handler
  read it, the job was counted as a job ahead of itself. The seed now reads
  the job's own place in the registry when it is already there.
