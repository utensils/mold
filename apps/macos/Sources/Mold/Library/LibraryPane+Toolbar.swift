import MoldClient
import SwiftUI

// The library's toolbar. Split from the pane purely for size.
extension LibraryPane {


    @ToolbarContentBuilder var toolbar: some ToolbarContent {
        ToolbarItem {
            Picker("Shelf", selection: $scope) {
                ForEach(LibraryScope.allCases) { shelf in
                    Label(shelf.title, systemImage: shelf.symbol).tag(shelf)
                }
            }
            .pickerStyle(.segmented)
            .labelStyle(.iconOnly)
            .help("All prints, favorites, or the trash")
        }
        ToolbarItem {
            Picker("Source", selection: $sourceHost) {
                Text("All machines").tag(MoldHost.ID?.none)
                ForEach(hosts.hosts) { host in
                    Text("\(host.name) (\(library.count(for: host.id)))")
                        .tag(MoldHost.ID?.some(host.id))
                }
            }
        }
        ToolbarItem {
            Slider(value: $edge, in: 88...260) { Text("Thumbnail size") }
                .frame(width: 110)
                .help("Thumbnail size")
        }
    }
}
