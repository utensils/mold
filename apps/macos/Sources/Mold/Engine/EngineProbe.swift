import Foundation

/// Waiting for the engine to actually be listening.
///
/// `mold_engine_start` returns as soon as the thread is spawned, so "started"
/// and "answering" are seconds apart on a cold home with a large gallery:
/// the tokio runtime, the DB migration, gallery-authority recovery and the
/// artifact-fact warming all precede the TCP bind (review 05-M3).
enum EngineProbe {
    enum Answer: Equatable {
        case answered
        case refused(String)
    }

    /// How long the engine is given to bind. A cold `MOLD_HOME` with a big
    /// library is the slow case, and it is minutes-adjacent rather than
    /// seconds; giving up early would report a working engine as broken.
    static let budget = Duration.seconds(120)
    static let interval = Duration.milliseconds(250)

    static func answer(
        port: UInt16,
        apiKey: String,
        budget: Duration = EngineProbe.budget,
        interval: Duration = EngineProbe.interval,
        ask: (UInt16, String) async -> Int? = EngineProbe.status
    ) async -> Answer {
        let attempts = max(1, Int(budget / interval))
        for _ in 0..<attempts {
            switch await ask(port, apiKey) {
            case 200:
                return .answered
            case 401, 403:
                // The engine is up and did not accept the key this launch
                // minted, which means it is not the engine we started.
                return .refused(
                    "Something else is already listening on 127.0.0.1:\(port) and it isn't "
                        + "this Mac's engine. Relaunch Mold to try again.")
            default:
                try? await Task.sleep(for: interval)
            }
        }
        return .refused(
            "The engine started but never answered on 127.0.0.1:\(port). Relaunch Mold to try "
                + "again — Mold's log in ~/Library/Logs/Mold has the detail.")
    }

    /// `nil` when nothing answered at all, which is the ordinary "not yet".
    private static func status(port: UInt16, apiKey: String) async -> Int? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/api/status") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        request.setValue(apiKey, forHTTPHeaderField: "X-Api-Key")
        guard let (_, response) = try? await URLSession.shared.data(for: request) else { return nil }
        return (response as? HTTPURLResponse)?.statusCode
    }
}
