import MoldClient
import SwiftUI

// The two bodies. Every bound is the HOST's -- the size range, the axes it
// accepts, the origins, and the turntable's own limits -- so a control here
// can never ask for something the server refuses.
extension MeshExportSheet {

    @ViewBuilder var geometryBody: some View {
        if let capabilities = prompt.capabilities {
            Toggle("Resize for printing", isOn: $scaled)
            if scaled {
                HStack {
                    Text("Longest side")
                    Slider(value: Binding(
                        get: { geometry.sizeMm ?? capabilities.sizeMm.default },
                        set: { geometry.sizeMm = $0 }),
                        in: capabilities.sizeMm.min...capabilities.sizeMm.max)
                    Text("\(Int((geometry.sizeMm ?? capabilities.sizeMm.default).rounded())) mm")
                        .monospacedDigit()
                        .frame(width: 70, alignment: .trailing)
                }
            }
            Picker("Up axis", selection: $geometry.upAxis) {
                ForEach(capabilities.upAxes, id: \.self) { axis in
                    Text(axis == .z ? "Z up" : "Y up").tag(axis)
                }
            }
            Picker("Origin", selection: $geometry.origin) {
                ForEach(capabilities.origins, id: \.self) { origin in
                    Text(origin == .floor ? "On the floor" : "Centred").tag(origin)
                }
            }
            Text(MeshExportGeometry.sizeLabel(bounds: prompt.bounds, options: resolved))
                .font(.callout)
                .foregroundStyle(.secondary)
        } else {
            // Absence of `capabilities.mesh.export_geometry` is the ONE gate:
            // an older host DROPS these keys rather than refusing them, so
            // offering knobs would promise a resize it never performs.
            Text("This machine writes the mesh in its stored units.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    /// What `resolved` would post, so the sentence and the request agree.
    var resolved: MeshExportGeometry {
        var value = geometry
        if !scaled { value.sizeMm = nil }
        return value
    }

    @ViewBuilder var turntableBody: some View {
        Stepper(value: $turntable.frames,
                in: MeshTurntableOptions.frameBounds, step: 4) {
            Text("\(turntable.frames) views around the mesh")
        }
        Stepper(value: $turntable.fps, in: 1...MeshTurntableOptions.maximumFPS) {
            Text("\(turntable.fps) frames a second")
        }
        Picker("Size", selection: $turntable.maxDimension) {
            ForEach([512, 1024, MeshTurntableOptions.maximumDimension], id: \.self) { edge in
                Text("\(edge) px").tag(edge)
            }
        }
        Toggle("Transparent background", isOn: $turntable.transparent)
        Text(duration).font(.callout).foregroundStyle(.secondary)
    }

    /// How long the clip will run, which is the thing the two steppers
    /// together decide and neither says on its own.
    private var duration: String {
        let seconds = Double(turntable.frames) / Double(max(turntable.fps, 1))
        return "\(String(format: "%.1f", seconds)) seconds a turn"
    }
}
