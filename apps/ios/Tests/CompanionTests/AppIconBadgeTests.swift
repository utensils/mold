import Testing
@testable import MoldCompanion

@MainActor struct AppIconBadgeTests {
    @Test func writesAreSerializedAndLatestCountWinsDuringPermissionRequest() async {
        var counts: [Int] = []
        var requests = 0
        var authorization: CheckedContinuation<Void, Never>?
        let badge = AppIconBadge(write: { counts.append($0) }, authorize: {
            requests += 1
            await withCheckedContinuation { authorization = $0 }
        })
        badge.update(2, allowPrompt: true)
        while authorization == nil { await Task.yield() }
        badge.update(1, allowPrompt: false)
        authorization?.resume()
        await badge.flush()
        #expect(counts == [1])
        badge.update(0, allowPrompt: false)
        await badge.flush()
        #expect(counts == [1, 0])
        #expect(requests == 1)
    }

    @Test func backgroundUpdatesAndZeroDoNotAskForPermission() async {
        var requests = 0
        var counts: [Int] = []
        let badge = AppIconBadge(write: { counts.append($0) }, authorize: { requests += 1 })
        badge.update(3, allowPrompt: false)
        await badge.flush()
        badge.update(0, allowPrompt: true)
        await badge.flush()
        #expect(requests == 0)
        #expect(counts == [3, 0])
    }
}
