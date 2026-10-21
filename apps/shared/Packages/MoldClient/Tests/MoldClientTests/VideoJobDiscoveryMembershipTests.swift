import Foundation
import Testing
@testable import MoldClient

struct VideoJobDiscoveryMembershipTests {
    let host = MoldHost(name:"Plato",baseURL:URL(string:"http://plato:7680")!,apiKey:"first")
    func snapshot(_ ids:[String],instance:String="installation",phase:String="queued",tick:Int=1) throws -> ActivityHostSnapshot {
        let rows = ids.map { "{\"id\":\"\($0)\",\"kind\":\"video_job\",\"phase\":\"\(phase)\",\"created_at_unix_ms\":1,\"updated_at_unix_ms\":\(tick),\"can_cancel\":true}" }
        let data = Data("{\"instance_id\":\"\(instance)\",\"observed_at_unix_ms\":\(tick),\"items\":[\(rows.joined(separator:","))],\"unavailable_kinds\":[]}".utf8)
        let value = try MoldJSON.decoder.decode(ActiveWorkSnapshot.self,from:data)
        return ActivityReconcile.host(routeURL:host.baseURL.absoluteString,previous:nil,result:.success(value))
    }
    @Test func newExternalParentTriggersDiscoveryButProgressAndOrderingDoNot() throws {
        let owner = VideoJobOwner(host:host,instanceId:"installation")
        let before = VideoJobDiscoveryMembership(owner:owner,activity:try snapshot(["first"]))
        let added = VideoJobDiscoveryMembership(owner:owner,activity:try snapshot(["first","external"]))
        #expect(before != added)
        let progressed = VideoJobDiscoveryMembership(owner:owner,activity:try snapshot(["external","first"],phase:"running",tick:2))
        #expect(added == progressed)
    }
    @Test func routeCredentialAndInstallationReplacementCancelOldDiscovery() throws {
        let old = VideoJobOwner(host:host,instanceId:"installation")
        let key = VideoJobDiscoveryMembership(owner:old,activity:try snapshot(["first"]))
        var changed = host; changed.apiKey = "replacement"
        #expect(key != VideoJobDiscoveryMembership(owner:.init(host:changed,instanceId:"installation"),activity:try snapshot(["first"])))
        changed = host; changed.baseURL = URL(string:"http://plato:7681")!
        #expect(key != VideoJobDiscoveryMembership(owner:.init(host:changed,instanceId:"installation"),activity:nil))
        let replacement = VideoJobDiscoveryMembership(owner:.init(host:host,instanceId:"replacement"),activity:try snapshot(["first"]))
        #expect(key != replacement)
        #expect(replacement.activeIDs.isEmpty, "an old installation snapshot cannot supply new parent's identities")
    }
}
