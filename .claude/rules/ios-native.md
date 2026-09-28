---
paths:
  - "apps/ios/**"
  - "apps/shared/**"
---

# Mold Studio Companion (apps/ios) and the shared Swift packages (apps/shared)

**What it is.** The native SwiftUI iPhone/iPad app: `io.utensils.mold.companion`, Home Screen label "Mold Studio", iOS 26+, remote-only. It is the macOS Mold Studio app's (`apps/macos`) little sibling and installs BESIDE the Tauri iPhone app (`apps/mobile`, `com.utensils.mold`), never replacing it. `apps/ios/docs/DESIGN.md` is the binding spec, `apps/ios/docs/PLAN.md` the milestone plan. Follow the Mac app, never the Tauri app's idioms (custom tab bars, toasts, fixed px sizes).

**Shared code.** `apps/shared/Packages/MoldClient` (wire + transport, Foundation only; `lint-layers` bans SwiftUI/AppKit/UIKit) and `MoldStyle` (tokens) serve BOTH apps; `platforms` declares macOS 26 and iOS 26. A change there must keep `make -C apps/macos lint` green and compile for iOS (`xcodebuild -scheme MoldClient -destination 'generic/platform=iOS Simulator' build` inside the package). Mac-only API goes behind `#if os(macOS)` (as `MoldHome` is). Credentials go through `CredentialStore`, keyed by host UUID: the Mac's `SecretStore` file on macOS, the Keychain on iOS; an empty key is a clear. Decision logic longer than ~10 lines that both apps need lands in MoldClient with its test, not twice in two app targets.

**Never register `mold://`.** The Tauri app owns it and iOS resolves a shared scheme arbitrarily. The companion's scheme is `moldstudio://`; desktop pairing QRs (`mold://pair?...`) are read by the in-app scanner and redeemed through `POST /api/pairing/claim`.

**Dynamic Type is a gate, not a goal.** Text styles only (`make lint` bans `.font(.system(size:`, `.custom`, `minimumScaleFactor`). Label/value rows ask `RowAxis.for(_:)` (stack from AX1). Secondary text is `.secondaryText`, never `.secondary` (the system colour is ~4.4:1 on white and fails the audit); the one filled button per screen uses `.prominentAction()` (`ProminentFill`). `make uitest` runs `performAccessibilityAudit` on every destination at xSmall, Large and AX5 in light AND dark; it may skip contrast only for elements under the bottom chrome. A new screen joins that audit.

**Extensions.** Widgets read the App Group snapshot only (no MoldClient, no URLSession, no Keychain); the Share extension stages into the App Group and never networks. `make lint` enforces both.

**Commands.** `make -C apps/ios gen|build|test|uitest|lint` (devshell: `companion-*`); pass `BUILD=/Volumes/ExternalStorage/...` locally to keep DerivedData off the internal disk. Simulator tests do not touch the desktop. Never run the MAC app's `make test` locally unasked: its host-app bundle launches Mold Studio on the user's desktop; CI (`macos-native.yml`) runs it. CI for this app: `.github/workflows/ios-native.yml`.
