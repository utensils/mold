import Foundation
import MoldClient
import SwiftUI
import Testing
import UserNotifications

@testable import Mold

@MainActor
struct NotificationActivationTests {
    @Test func theBuiltAppProhibitsMultipleInstances() {
        #expect(Bundle.main.object(forInfoDictionaryKey: "LSMultipleInstancesProhibited") as? Bool == true)
    }

    @Test func coldLaunchClicksAreAcceptedThenDeliveredOnceInOrder() {
        let responses = NotificationResponses()
        let host = UUID()
        var completions = 0
        responses.receive(action: UNNotificationDefaultActionIdentifier,
                          userInfo: ["kind": "print", "host": host.uuidString, "filename": "new.png"]) {
            completions += 1
        }
        responses.receive(action: UNNotificationDefaultActionIdentifier, userInfo: ["kind": "failure"]) {
            completions += 1
        }
        #expect(completions == 2)
        var received: [NotificationRoute] = []
        responses.install { received.append($0) }
        #expect(received == [.openPrint(PrintID(host: host, filename: "new.png")), .openQueue])
        responses.install { received.append($0) }
        #expect(received.count == 2)
    }

    @Test func warmClicksAreDeliveredBeforeCompletionAndUseTheLatestHandler() {
        let responses = NotificationResponses()
        var events: [String] = []
        responses.install { _ in events.append("old") }
        responses.install { _ in events.append("route") }
        responses.receive(action: UNNotificationDefaultActionIdentifier, userInfo: ["kind": "failure"]) {
            events.append("complete")
        }
        #expect(events == ["route", "complete"])
    }

    @Test func dismissCustomAndMalformedResponsesCompleteWithoutNavigation() {
        let responses = NotificationResponses()
        var completions = 0
        var received: [NotificationRoute] = []
        for action in [UNNotificationDismissActionIdentifier, "unknown", ""] {
            responses.receive(action: action, userInfo: ["kind": "failure"]) { completions += 1 }
        }
        for payload in [[:], ["kind": "print"], ["kind": "print", "host": "bad", "filename": "a.png"]] {
            responses.receive(action: UNNotificationDefaultActionIdentifier, userInfo: payload) {
                completions += 1
            }
        }
        responses.install { received.append($0) }
        #expect(received.isEmpty)
        #expect(completions == 6)
    }

    @Test func aPrintClickLeavesTheCollectionAndPreservesItsHostIdentity() {
        let defaults = UserDefaults(suiteName: "notification-activation-\(UUID())")!
        let navigation = LibraryNavigation(defaults: defaults)
        navigation.scope = .collection(slug: "other")
        var destination = Destination.generate
        let binding = Binding(get: { destination }, set: { destination = $0 })
        let printID = PrintID(host: UUID(), filename: "new.png")
        applyNotificationRoute(.openPrint(printID), destination: binding, navigation: navigation)
        #expect(destination == .library)
        #expect(navigation.scope == .all)
        #expect(navigation.reveal == printID)
        applyNotificationRoute(.openQueue, destination: binding, navigation: navigation)
        #expect(destination == .queue)
    }
}
