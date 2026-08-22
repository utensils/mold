# Android App

Mold for Android is the remote-only Mold Studio client. It connects to Mold
servers over LAN, Tailscale, or HTTPS and shares the same Create, Library,
Models, Machines, and Settings surface as the iPhone app. Model execution stays
on the remote machine; per-host API keys are encrypted by an app-owned,
non-exportable Android Keystore key.

Android development builds and the current nightly APK are documented in the
[mobile maintainer guide](https://github.com/utensils/mold/blob/main/apps/mobile/README.md#android).
Google Play publishing remains release work.

## Connect a machine

Open **Machines** and scan the one-use pairing code shown by Mold Studio on the
server, discover `_mold._tcp` machines on the local network, or enter a LAN,
DNS, HTTPS, or Tailscale MagicDNS address manually. Pairing transfers a
two-minute one-use ticket, not the durable API key.

## Identity photos (PuLID)

An identity photo conditions a print on a person's face while the prompt owns
the scene, styling, and composition. The **Identity** well appears in the main
Create form beside the source wells only when the selected model and target
machine positively advertise identity support. It stays hidden for Sequence.

Tap the well and choose one native action:

- **Choose photo** opens Android's Photo Picker. Mold requests no broad photo or
  storage permission.
- **Take photo** opens the installed camera through a temporary content URI.

Use a PNG or JPEG no larger than 16 MiB, 8192 px per side, or 32 MP. Android
checks the provider-reported file size before reading the photo, and the bytes
are sent verbatim rather than cropped or fitted to the canvas. The source sheet
uses 48dp controls and the Android back gesture dismisses it without changing
the draft.

Switching to a model without identity support parks the attached photo: the
well disappears, the identity fields stay off the request, and **Develop**
remains available. Returning to a qualified model restores the well and photo.
**Identity strength** and **Identity start step** live in Advanced, count toward
its badge, and clear with Advanced Reset while the photo remains attached.

Identity requests cannot combine with a LoRA or img2img source. Every refusal
is shown inline and queues nothing. Under **Auto** or **Most capable**, Mold
considers only machines whose own model row advertises identity support and
does not use the legacy placement fallback. Prepared Batch siblings inherit the
same photo and controls.

Library **Info** shows the photo name, short SHA-256, effective strength, and
start step. **Use as prompt** restores the controls and re-attaches the exact
photo from the device stash; if the bytes are gone, Create reports that miss
inline instead of silently using another face. Saved metadata never contains
the photo bytes.

See [Identity Photos (PuLID)](/guide/generating#identity-photos-pulid) for model
requirements, limits, and the one-time InsightFace license acceptance.
