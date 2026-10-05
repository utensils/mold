# Native iOS permissions

Mold Studio requests access when the person uses the related feature. Denied
access offers a native alert with **Open Settings** and **Not Now**, using Apple's
public app-settings URL (notification recovery uses the notification-settings
URL). Return to the app and retry saving or taking a photo; scanning, discovery
and Settings refresh on activation. A restriction explains Screen Time/device
management without promising that a Settings toggle can repair it.

| Access | Reason and request | Recovery |
| --- | --- | --- |
| Photos, add only | Save a picture/video, or enable Save Finished Prints to Photos | Native denial alert before downloading files; Settings also offers recovery. Auto-save never requests access or interrupts a finished render. |
| Camera, video capture authorization | Take Photo or open supported QR scanner | Request before presenting capture; denied access offers Settings. The scanner retains pasted-link fallback. No microphone recording. |
| Local Network | Browse `_mold._tcp` or connect to a nearby machine | Bonjour policy denial (`-65570`) offers Settings in Machines/Nearby, including with no machines. Other network failures describe connectivity. Browsing restarts after returning from Settings. |
| Notifications | First submitted render, or Enable Notifications in Settings | Denied Settings section offers notification-settings recovery and refreshes on return. App notification-kind toggles remain separate preferences. |
| Live Activities | Follow renders on the Lock Screen; ActivityKit controls authorization | Phone Settings offers app-settings recovery when system authorization is disabled; no extra fake system request. |

Photo imports use PhotosPicker: only the selected items are delivered, without
requesting broad library access. Files imports use scoped document-picker URLs.
Share extension input is delivered by the system and staged in the App Group.
Keychain/App Group/associated-domain entitlements are capabilities, not user
permission prompts. Background refresh is system scheduling, not a requestable
privacy authorization. No location, contacts, tracking, Bluetooth or microphone
permission is required by the native companion's current features.

## Video-save crash

PhotoKit executes its change block on an arbitrary serial queue. The previous
block inherited MainActor isolation from PrintActions and crashed with
`_dispatch_assert_queue_fail` / `_swift_task_checkIsolatedSwift` after authorization.
PhotosWriter builds an explicitly nonisolated, Sendable callback, capturing only
immutable URL/video descriptors resolved on MainActor. Export files remain alive
until the awaited Photos operation completes, then are removed. Video resources
use container format (MP4/MOV/M4V), falling back to the filename extension for
older hosts without `format`. Animated GIF/WebP/APNG use photo resources,
independently of the viewer's broader playback-kind classification.

## Apple references

- [Privacy guidance](https://developer.apple.com/design/human-interface-guidelines/privacy/)
- [Requesting access](https://developer.apple.com/documentation/uikit/requesting-access-to-protected-resources)
- [App Settings URL](https://developer.apple.com/documentation/uikit/uiapplication/opensettingsurlstring)
- [Notification Settings URL](https://developer.apple.com/documentation/uikit/uiapplication/opennotificationsettingsurlstring)
- [Photos change-block queue](https://developer.apple.com/documentation/photos/phphotolibrary/performchanges(_:completionhandler:))
- [Photos picker privacy](https://developer.apple.com/videos/play/wwdc2020/10652/)
- [Camera authorization](https://developer.apple.com/documentation/avfoundation/requesting-authorization-to-capture-and-save-media)
- [Local Network policy denial](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy)
- [Live Activity authorization](https://developer.apple.com/documentation/activitykit/activityauthorizationinfo/areactivitiesenabled)

## Validation

PermissionRecoveryTests cover denied/restricted/granted/not-yet-requested states,
public Settings destinations and Local Network error classification. Native UI
regressions save a real H.264 MP4 to Simulator Photos, reject Photos access,
cancel recovery, retry denied access, and open Settings. The permission tests
live in the existing LibraryViewerTests CI shard. Hardware camera capture,
VisionKit QR scanning, Screen Time/MDM restrictions, and Local Network privacy
must also be checked on an iPhone; Simulator does not prove those behaviors.
