package com.utensils.mold.mobile_native

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PairingScanSessionTest {
    @Test
    fun cancel_returns_pending_scan_and_invalidates_late_camera_callbacks() {
        val session = PairingScanSession<String>()
        val id = session.begin("pending invoke")

        val cancelled = session.cancel()

        assertEquals("pending invoke", cancelled?.value)
        assertFalse(session.isActive(id))
        assertNull(session.complete(id))
    }

    @Test
    fun successful_scan_settles_once_and_allows_another_scan() {
        val session = PairingScanSession<String>()
        val first = session.begin("first")

        assertEquals("first", session.complete(first))
        assertNull(session.complete(first))
        val second = session.begin("second")
        assertTrue(session.isActive(second))
    }
}
