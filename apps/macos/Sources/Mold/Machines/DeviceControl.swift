import MoldClient

/// What a machine will let you do to one of its cards.
///
/// TWO flags, never one. `devices.lifecycle` says the route is there;
/// `dispatch.v2_authoritative` says the runtime that answers it owns dispatch
/// and will honour the answer. A legacy, observe, CPU-fallback or maintenance
/// runtime reports the first and not the second, and persisting a change it
/// cannot enforce would be a lie told in a switch.
enum DeviceControl: Equatable {
    /// A live switch this machine will act on now.
    case live(isOn: Bool, isEnabled: Bool)
    /// No live authority, but a card that is off can be turned on for the
    /// next start. A different power, not a weaker one.
    case enableAtRestart
    /// Nothing to offer. The row still says what the card IS.
    case readOnly
}

extension DeviceControl {
    /// The rules, in order. `capabilities == nil` -- a machine that has not
    /// answered yet -- falls through to `.readOnly` by construction, which is
    /// what an unknown host should get.
    static func resolve(_ device: DeviceInfo,
                        on capabilities: Capabilities?,
                        isChanging: Bool) -> DeviceControl {
        // The server refuses `enabled: true` with 409 DEVICE_STARTUP_EXCLUDED
        // whatever the dispatch mode, because the card was excluded by startup
        // selection. Offering either control here offers a 409.
        if device.adminState == .startupExcluded { return .readOnly }

        if capabilities?.canChangeDeviceLifecycle == true,
           capabilities?.dispatchIsAuthoritative == true {
            // A card mid-transition has already been asked; asking again is
            // not a second answer, it is a second request.
            let settled = device.adminState != .starting && device.adminState != .draining
            return .live(isOn: device.desiredEnabled, isEnabled: !isChanging && settled)
        }

        // Only when the card is OFF -- "Enable at next restart" on a card
        // that is already on is a button that does nothing.
        if capabilities?.canEnableDeviceAtRestart == true, !device.desiredEnabled {
            return .enableAtRestart
        }

        return .readOnly
    }
}
