import Foundation
import MoldClient

// Re-reading ONE machine's prints.
//
// `refresh()` asks every machine, which is right when the pane is opened or
// somebody presses Refresh. It is not right after a single machine published
// a single print: desktop re-reads only the owning host
// (`gallery.refreshHost(authority.sourceKey)`), and on a three-machine fleet
// the fleet-wide read is three listings for one finished clip.
@MainActor
extension LibraryStore {
    func refresh(on host: MoldHost.ID) async {
        guard let machine = hosts.host(host) else { return }
        let client = hosts.backend(for: machine)
        let etag = etags[host]
        let result: Result<Fetched<[GalleryPrint]>, Error>
        do {
            result = .success(try await client.gallery(etag: etag))
        } catch {
            result = .failure(error)
        }
        apply(result, for: machine)
        rebuild()
    }
}
