import Foundation
import MoldClient

/// Everything one launch of the in-process engine needs resolved BEFORE the
/// first FFI call, and the refusals that stop it starting wrong.
///
/// Separate from `MoldEngine` because `start()` lives behind
/// `#if MOLD_EMBEDDED_ENGINE` and a remote-only build cannot call it: the
/// decisions that must be right -- which home, which key, and whether to
/// refuse -- are here, where every build can test them.
struct EngineLaunch: Equatable, Sendable {
    let home: String
    /// Never `nil`. A keyless engine on loopback is what review 05-H1 is
    /// about: `POST /api/shutdown` is unauthenticated by design because the
    /// caller is loopback, and with no key the rest of the API is too.
    let apiKey: String
    let logDirectory: String
}

/// Why the engine will not be started. Each case carries the sentence shown
/// in Settings ▸ This Mac -- the app never invents a second wording for the
/// same refusal.
enum EngineLaunchRefusal: Error, Equatable {
    case home(String)
    case key(String)

    var reason: String {
        switch self {
        case let .home(reason): reason
        case let .key(reason): reason
        }
    }
}

enum EngineLaunchPlan {
    /// Resolves a launch, or refuses.
    ///
    /// The key follows `SecretStore.localEngineAPIKey`'s precedence, which is
    /// `secrets.rs:103-119`'s. A failure to PERSIST a freshly minted key is a
    /// refusal and not a shrug: starting anyway would either mint a new key
    /// every launch (so "This Mac" holds one the engine does not accept) or
    /// start keyless, which is the hole itself.
    static func resolve(
        home: MoldHome,
        secrets: SecretStore,
        logDirectory: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> EngineLaunch {
        if let reason = home.unavailableReason { throw EngineLaunchRefusal.home(reason) }
        let key: String
        do {
            key = try secrets.localEngineAPIKey(environment: environment)
        } catch {
            throw EngineLaunchRefusal.key(
                "Mold couldn't save this Mac's engine key to "
                    + "\(secrets.directory.path(percentEncoded: false)). The engine needs one: "
                    + "without it, any web page that finds its port can read and delete your library."
            )
        }
        return EngineLaunch(
            home: home.url.path(percentEncoded: false),
            apiKey: key,
            logDirectory: logDirectory
        )
    }
}
