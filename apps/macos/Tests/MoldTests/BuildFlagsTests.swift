import Testing

/// A Debug build must define `DEBUG`.
///
/// It once did not: `make engine` writes `SWIFT_ACTIVE_COMPILATION_CONDITIONS`
/// into `Engine.xcconfig`, xcodegen drops its Debug preset for any key the
/// config file defines, and `DEBUG` vanished on every Mac with the engine
/// linked. Every `#if DEBUG` hook compiled out and every `#if DEBUG` test
/// passed by running nothing, so the gates stayed green. This test is
/// deliberately NOT wrapped in `#if DEBUG` -- it is the one that must run.
struct BuildFlagsTests {
    @Test func aDebugBuildDefinesDEBUG() {
        var defined = false
        #if DEBUG
        defined = true
        #endif
        // `-Onone` is how a Debug configuration compiles; Release is `-O`.
        #expect(defined == _isDebugAssertConfiguration())
    }
}
