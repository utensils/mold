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

    /// The contextual menu's twin of whatever control this row draws --
    /// `nil` where the row draws none, so a right click offers nothing
    /// rather than something disabled.
    ///
    /// `named` is the row's own title, so the item reads as the switch's
    /// accessibility name does ("Use NVIDIA L40S  #0"). A card mid-transition
    /// is `.live(_, isEnabled: false)`: it has already been asked, and asking
    /// again is a second request, not a second answer.
    /// The card's one control as a menu row, or none at all -- a machine
    /// this app cannot change offers nothing rather than a disabled item.
    /// The kind is what the row would ask the machine FOR.
    func menu(named name: String) -> [RowAction<Bool>] {
        switch self {
        case let .live(isOn, isEnabled):
            guard isEnabled else { return [] }
            return isOn
                ? [RowAction(kind: false, title: "Stop Using \(name)")]
                : [RowAction(kind: true, title: "Use \(name)")]
        case .enableAtRestart:
            return [RowAction(kind: true, title: "Enable at next restart")]
        case .readOnly:
            return []
        }
    }
}
