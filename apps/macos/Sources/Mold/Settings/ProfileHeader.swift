import MoldClient
import SwiftUI

/// The active profile, the rest of the list, and one sentence about why
/// neither is a control -- switching writes the active-profile row without
/// touching the running server's loaded config (fact 6), so a table that
/// could switch would show one profile's values while an edit landed in
/// another's (design decision 9). Absent entirely on a machine that answered
/// 503 -- there is nothing to say about a database that is off.
struct ProfileHeader: View {
    let profiles: ConfigProfiles?

    var body: some View {
        if let profiles {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text("Profile").foregroundStyle(.secondary)
                    Text(profiles.active).fontWeight(.medium)
                    if others(profiles).isEmpty == false {
                        Text("(also: \(others(profiles).joined(separator: ", ")))")
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.callout)
                Text("""
                     Profiles are switched with `mold config profile` on the machine \
                     itself. A live server keeps the settings it loaded when it started, \
                     so switching here would change where an edit lands without changing \
                     what this table shows.
                     """)
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    private func others(_ profiles: ConfigProfiles) -> [String] {
        profiles.profiles.filter { $0 != profiles.active }
    }
}
