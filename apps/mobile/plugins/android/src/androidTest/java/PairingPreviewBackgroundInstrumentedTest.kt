package com.utensils.mold.mobile_native

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.webkit.WebView
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertSame
import org.junit.Assert.assertEquals
import org.junit.Test

class PairingPreviewBackgroundInstrumentedTest {
    @Test
    fun cancellationThenDisposalRestoresBackgroundOnlyOnce() {
        val context: Context = ApplicationProvider.getApplicationContext()
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            val view = WebView(context)
            try {
                val original = ColorDrawable(Color.BLUE)
                view.background = original
                val background = PairingPreviewBackground(view)
                background.restore() // Attachment failed before capturing anything.
                assertSame(original, view.background)
                background.makeTransparent()
                assertEquals(Color.BLUE, original.color)
                background.makeTransparent()
                background.restore() // User cancels.
                assertSame(original, view.background)
                val replacement = ColorDrawable(Color.WHITE)
                view.background = replacement
                background.restore() // Activity destruction must not erase newer state.
                assertSame(replacement, view.background)
                background.makeTransparent()
                background.restore()
                assertSame(replacement, view.background)
            } finally {
                view.destroy()
            }
        }
    }
}
