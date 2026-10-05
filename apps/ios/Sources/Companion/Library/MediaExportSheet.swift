import MoldClient
import SwiftUI

struct MediaExportSheet: View {
    @Bindable var session: MediaExportSession
    @Environment(PrintActions.self) private var actions
    var body: some View {
        NavigationStack {
            Form {
                if session.loading { ProgressView("Reading export options…") }
                if let error = session.error {
                    Section { Text(error).foregroundStyle(.secondaryText)
                        if session.options == nil { Button("Retry") { session.load() } }
                    }
                }
                if session.options != nil && !session.formats.isEmpty {
                    Section("Format") {
                        MediaExportPicker("Format", value: session.format.uppercased(), selection: $session.format, identifier: "export-format") {
                            ForEach(session.formats, id: \.self) { Text($0.uppercased()).tag($0) }
                        }
                    }
                    if session.animation { animationOptions }
                    else if session.geometry != nil { geometryOptions }
                    Section("Destination") {
                        MediaExportPicker("Destination", value: session.destination.title, selection: $session.destination, identifier: "export-destination") {
                            ForEach(session.destinations) { Text($0.title).tag($0) }
                        }
                    }
                    Section {
                        Button(session.converting ? "Converting…" : "Export") { session.submit() }
                            .prominentAction().disabled(!session.valid)
                            .accessibilityIdentifier("export-submit")
                    }
                }
            }
            .accessibilityIdentifier("export-form")
            .disabled(session.converting)
            .navigationTitle("Export Media")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { session.cancel(); actions.sheet = nil }
                        .accessibilityIdentifier("export-cancel")
                }
                ToolbarItemGroup(placement: .keyboard) { Spacer(); Button("Done") { hideKeyboard() } }
            }
        }
        .permissionAlert(Binding(get: { actions.permissionRecovery }, set: { actions.permissionRecovery = $0 }))
        .task { session.load() }
        .onChange(of: session.format) { session.selectFormat() }
        .onChange(of: session.frameLimit) { session.frames = min(session.frames, session.frameLimit) }
        .interactiveDismissDisabled(session.converting)
    }

    @ViewBuilder private var animationOptions: some View {
        if session.isGif {
            Section("Playback") {
                MediaExportPicker("Playback", value: session.playback == .loop ? "Loop" : "Bounce", selection: $session.playback, identifier: "export-playback") {
                    ForEach(session.options?.playbackChoices ?? [], id: \.self) {
                        Text($0 == .loop ? "Loop" : "Bounce").tag($0)
                    }
                }
                MediaExportPicker("Repeat", value: session.repeatMode == .forever ? "Forever" : "Once", selection: $session.repeatMode, identifier: "export-repeat") {
                    ForEach(session.options?.repeatChoices ?? [], id: \.self) {
                        Text($0 == .forever ? "Forever" : "Once").tag($0)
                    }
                }
                Text("Bounce plays forward, then reverses.").foregroundStyle(.secondaryText)
            }
            if session.takesPause, let control = session.pauseControl {
                Section(session.playback == .bounce ? "Pause at turns" : "Pause between loops") {
                    TextField("Pause in milliseconds", text: $session.pauseText)
                        .keyboardType(.numberPad).accessibilityIdentifier("export-pause")
                    Slider(value: Binding(get: { Double(session.pauseText) ?? 0 },
                        set: { session.pauseText = String(Int($0)) }), in: Double(control.min)...Double(control.max), step: Double(control.step))
                        .accessibilityLabel("Pause in milliseconds")
                    Button("No pause (0 ms)") { session.pauseText = "0" }
                    Text("0 adds no pause and keeps the frame rate.").foregroundStyle(.secondaryText)
                }
            }
        }
        Section("Size and frame rate") {
            MediaExportPicker("Longest side", value: session.maxDimension == 0 ? "Original" : "\(session.maxDimension) px", selection: $session.maxDimension, identifier: "export-size") {
                if session.kind == .video { Text("Original").tag(0) }
                ForEach(session.kind == .mesh ? [1024, 720, 512, 480] : [1080, 720, 480], id: \.self) {
                    Text("\($0) px").tag($0)
                }
            }
            MediaExportPicker("Frame rate", value: session.fps == 0 ? "Original" : "\(session.fps) fps", selection: $session.fps, identifier: "export-fps") {
                if session.kind == .video { Text("Original").tag(0) }
                ForEach(session.kind == .mesh ? [24, 12, 10, 8] : [24, 12, 8], id: \.self) { Text("\($0) fps").tag($0) }
            }
            if session.kind == .mesh {
                Stepper("Views per turn: \(session.frames)", value: $session.frames, in: 8...session.frameLimit)
                    .accessibilityIdentifier("export-frames")
                Toggle("Transparent background", isOn: $session.transparent).accessibilityIdentifier("export-transparent")
                Text(session.isGif ? "GIF has a hard transparent edge." : "Keeps the object's soft outline.")
                    .foregroundStyle(.secondaryText)
                if session.frameLimit < 180 {
                    Text("This size and background allow up to \(session.frameLimit) views.").foregroundStyle(.secondaryText)
                }
            }
        }
    }

    @ViewBuilder private var geometryOptions: some View {
        if let caps = session.meshCapabilities?.exportGeometry {
            Section("Geometry") {
                Toggle("Use format default", isOn: Binding(get: { session.geometry?.sizeMm == nil }, set: {
                    session.geometry?.sizeMm = $0 ? nil : caps.sizeMm.default
                })).accessibilityIdentifier("export-size-default")
                if session.geometry?.sizeMm != nil {
                    AdaptiveRow {
                        Text("Longest side in mm")
                    } value: {
                        TextField("", value: Binding(get: { session.geometry?.sizeMm ?? caps.sizeMm.default },
                            set: { session.geometry?.sizeMm = $0 }), format: .number)
                            .keyboardType(.decimalPad).accessibilityLabel("Longest side in millimeters")
                            .accessibilityIdentifier("export-size-mm")
                    }
                }
                MediaExportPicker("Up axis", value: (session.geometry?.upAxis ?? .y).rawValue.uppercased(), selection: Binding(get: { session.geometry?.upAxis ?? .y }, set: { session.geometry?.upAxis = $0 }), identifier: "export-axis") {
                    ForEach(caps.upAxes, id: \.self) { Text($0.rawValue.uppercased()).tag($0) }
                }
                MediaExportPicker("Origin", value: session.geometry?.origin == .floor ? "Floor" : "Center", selection: Binding(get: { session.geometry?.origin ?? .floor }, set: { session.geometry?.origin = $0 }), identifier: "export-origin") {
                    ForEach(caps.origins, id: \.self) { Text($0 == .floor ? "Floor" : "Center").tag($0) }
                }
            }
        }
    }
    private func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

/// A native checkmarked menu with an explicit wrapping selected-value label.
/// The standard Form Picker can truncate even after stacking at AX5.
private struct MediaExportPicker<Selection: Hashable, Choices: View>: View {
    let title: String
    let value: String
    @Binding var selection: Selection
    let identifier: String
    let choices: Choices
    init(_ title: String, value: String, selection: Binding<Selection>, identifier: String, @ViewBuilder choices: () -> Choices) {
        self.title = title; self.value = value; _selection = selection
        self.identifier = identifier; self.choices = choices()
    }
    var body: some View {
        AdaptiveRow {
            Text(title)
        } value: {
            Menu {
                Picker(title, selection: $selection) { choices }
            } label: {
                HStack {
                    Text(value).fixedSize(horizontal: false, vertical: true)
                    Image(systemName: "chevron.up.chevron.down").font(.caption).accessibilityHidden(true)
                }
                .frame(minWidth: 44, minHeight: 44)
            }
            .accessibilityLabel("\(title), \(value)")
            .accessibilityIdentifier(identifier)
        }
    }
}
