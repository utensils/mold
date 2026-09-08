package com.utensils.mold.mobile_native

import android.content.Context
import android.webkit.WebView
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Test

class AndroidTextScaleInstrumentedTest {
    @Test
    fun appliesInitialAndChangedTextZoomWithoutChangingZoomSupport() {
        val context: Context = ApplicationProvider.getApplicationContext()
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            val view = WebView(context)
            try {
                val initialPageZoom = view.settings.supportZoom()
                applyAndroidTextScale(view, 1.3f)
                assertEquals(130, view.settings.textZoom)
                applyAndroidTextScale(view, 2f)
                assertEquals(200, view.settings.textZoom)
                applyAndroidTextScale(view, 1f)
                assertEquals(100, view.settings.textZoom)
                assertEquals(initialPageZoom, view.settings.supportZoom())
            } finally {
                view.destroy()
            }
        }
    }
}
