import MoldClient
import SwiftUI

/// Shared by clip conversion and mesh turntables; absent pause capability is an older host.
struct GifExportSelection {
    var playback: GifPlayback
    var repeatMode: GifRepeat
    var pauseText: String

    init(options: ExportOptions?) {
        playback = options?.playbackChoices.first ?? .loop
        repeatMode = options?.repeatChoices.first ?? .forever
        pauseText = String(Self.control(options)?.defaultValue ?? 0)
    }

    static func control(_ options: ExportOptions?) -> GifPauseControl? {
        options?.gifPause.flatMap { $0.valid ? $0 : nil }
    }

    func takesPause(format: String, options: ExportOptions?) -> Bool {
        format == "gif" && (playback == .bounce || repeatMode == .forever) && Self.control(options) != nil
    }

    func pause(format: String, options: ExportOptions?) -> Int? {
        takesPause(format: format, options: options) ? Int(pauseText) : nil
    }

    func valid(format: String, options: ExportOptions?) -> Bool {
        guard format == "gif" else { return true }
        if let options {
            guard options.playbackChoices.contains(playback), options.repeatChoices.contains(repeatMode) else { return false }
        }
        guard takesPause(format: format, options: options) else { return true }
        guard let value = Int(pauseText) else { return false }
        return Self.control(options)?.accepts(value) == true
    }
}

struct GifExportControls: View {
    @Binding var selection: GifExportSelection
    let options: ExportOptions?

    var body: some View {
        if let options {
            Picker("Playback", selection: $selection.playback) {
                ForEach(options.playbackChoices, id: \.self) { choice in
                    Text(choice == .loop ? "Loop" : "Bounce").tag(choice)
                }
            }
            .help("Loop plays forward; Bounce plays forward and then backward")
            Picker("Repeat", selection: $selection.repeatMode) {
                ForEach(options.repeatChoices, id: \.self) { choice in
                    Text(choice == .forever ? "Forever" : "Once").tag(choice)
                }
            }
            .help("Choose whether the animation plays once or keeps repeating")
        }
        if selection.takesPause(format: "gif", options: options), let control = GifExportSelection.control(options) {
            TextField(selection.playback == .bounce ? "Pause at turns (ms)" : "Pause between loops (ms)", text: $selection.pauseText)
                .help("Add a pause at each turn or repeat. Enter 0 for no extra pause; 1000 milliseconds is one second.")
            if control.min < control.max {
            Slider(value: Binding(get: { Double(selection.pauseText) ?? Double(control.defaultValue) },
                                  set: { selection.pauseText = String(Int($0)) }),
                   in: Double(control.min)...Double(control.max), step: Double(control.step))
            }
            Button("No pause (0 ms)") { selection.pauseText = "0" }
                .disabled(!control.accepts(0))
            if !selection.valid(format: "gif", options: options) {
                Text("Enter a pause from \(control.min) to \(control.max) ms in steps of \(control.step).")
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
    }
}
