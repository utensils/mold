import AVFAudio

enum PlaybackAudio {
    /// AVPlayer activates the session when playback starts. Merely browsing
    /// the library must not interrupt another app's music.
    static func configure(_ session: AVAudioSession = .sharedInstance()) throws {
        try session.setCategory(.playback, mode: .moviePlayback)
    }
}
