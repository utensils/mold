package com.utensils.mold.mobile_native

import android.webkit.WebView
import kotlin.math.roundToInt

/** Follow Android's text preference without scaling the entire WebView viewport. */
fun applyAndroidTextScale(webView: WebView, fontScale: Float) {
    val scale = fontScale.takeIf { it.isFinite() && it > 0f } ?: 1f
    webView.settings.textZoom = (scale * 100).roundToInt().coerceAtLeast(1)
}
