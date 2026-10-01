import SwiftUI
import UIKit

/// Start sideways to sweep; a vertical start remains the scroll view's pan.
/// Once acquired, the finger may move through rows and pause at an edge.
struct LibrarySelectionRecognizer: UIGestureRecognizerRepresentable {
    let enabled: Bool
    let canStart: (CGPoint) -> Bool
    let changed: (CGPoint, CGPoint) -> Void
    let ended: () -> Void

    func makeCoordinator(converter: CoordinateSpaceConverter) -> Coordinator {
        Coordinator(converter: converter, owner: self)
    }

    func makeUIGestureRecognizer(context: Context) -> UIPanGestureRecognizer {
        let pan = UIPanGestureRecognizer()
        pan.maximumNumberOfTouches = 1
        pan.delegate = context.coordinator
        pan.isEnabled = enabled
        return pan
    }

    func updateUIGestureRecognizer(_ recognizer: UIPanGestureRecognizer, context: Context) {
        context.coordinator.owner = self
        recognizer.isEnabled = enabled
        if !enabled { context.coordinator.finish() }
    }

    func handleUIGestureRecognizerAction(_ recognizer: UIPanGestureRecognizer, context: Context) {
        let coordinator = context.coordinator
        switch recognizer.state {
        case .began: coordinator.begin(recognizer)
        case .changed: coordinator.emit()
        case .ended, .cancelled, .failed: coordinator.finish()
        default: break
        }
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        let converter: CoordinateSpaceConverter
        var owner: LibrarySelectionRecognizer
        var start: CGPoint?
        weak var scroll: UIScrollView?
        private var task: Task<Void, Never>?

        deinit { task?.cancel() }

        init(converter: CoordinateSpaceConverter, owner: LibrarySelectionRecognizer) {
            self.converter = converter
            self.owner = owner
        }

        func gestureRecognizerShouldBegin(_ gestureRecognizer: UIGestureRecognizer) -> Bool {
            guard owner.enabled, let pan = gestureRecognizer as? UIPanGestureRecognizer else { return false }
            let translation = converter.translation(in: .global) ?? .zero
            let point = converter.location(in: .global)
            let origin = CGPoint(x: point.x - translation.x, y: point.y - translation.y)
            let velocity = pan.velocity(in: pan.view)
            return abs(velocity.x) > abs(velocity.y) && owner.canStart(origin)
        }

        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                               shouldBeRequiredToFailBy other: UIGestureRecognizer) -> Bool {
            // Give the sweep first refusal. It fails immediately for vertical
            // motion, allowing scrolling and pull-to-refresh to proceed.
            other is UIPanGestureRecognizer && other.view is UIScrollView
        }

        func begin(_ pan: UIPanGestureRecognizer) {
            let point = converter.location(in: .global)
            let translation = converter.translation(in: .global) ?? .zero
            start = CGPoint(x: point.x - translation.x, y: point.y - translation.y)
            // The hit view can be deep in SwiftUI's hosting hierarchy.
            var hit = pan.view?.window?.hitTest(pan.location(in: pan.view?.window), with: nil)
            while let view = hit {
                if let candidate = view as? UIScrollView { scroll = candidate; break }
                hit = view.superview
            }
            emit()
            task = Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    do { try await Task.sleep(for: .milliseconds(33)) } catch { return }
                    guard let self else { return }
                    self.tick()
                }
            }
        }

        func emit() {
            guard let start else { return }
            var point = converter.location(in: .global)
            if let scroll {
                let frame = scroll.convert(scroll.bounds, to: nil)
                point.y = min(frame.maxY - scroll.adjustedContentInset.bottom - 1,
                              max(frame.minY + scroll.adjustedContentInset.top + 1, point.y))
            }
            owner.changed(point, start)
        }

        private func tick() {
            guard owner.enabled, let scroll, start != nil, scroll.window != nil else {
                finish(); return
            }
            let point = converter.location(in: .global)
            let frame = scroll.convert(scroll.bounds, to: nil)
            let top = frame.minY + scroll.adjustedContentInset.top
            let bottom = frame.maxY - scroll.adjustedContentInset.bottom
            let margin: CGFloat = 56
            let step: CGFloat
            if point.y < top + margin { step = -min(12, max(0, (top + margin - point.y) / 4)) }
            else if point.y > bottom - margin { step = min(12, max(0, (point.y - bottom + margin) / 4)) }
            else { step = 0 }
            let minimum = -scroll.adjustedContentInset.top
            let maximum = max(minimum, scroll.contentSize.height - scroll.bounds.height + scroll.adjustedContentInset.bottom)
            let offset = min(maximum, max(minimum, scroll.contentOffset.y + step))
            if offset != scroll.contentOffset.y {
                scroll.setContentOffset(CGPoint(x: scroll.contentOffset.x, y: offset), animated: false)
            }
            emit()
        }

        func finish() {
            task?.cancel()
            task = nil
            scroll = nil
            guard start != nil else { return }
            start = nil
            owner.ended()
        }
    }
}
