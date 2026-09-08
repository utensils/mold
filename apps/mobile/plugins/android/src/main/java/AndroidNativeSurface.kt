package com.utensils.mold.mobile_native

import android.app.Activity
import android.content.res.Configuration
import android.graphics.Color
import android.os.Build
import android.webkit.WebView
import androidx.core.view.WindowCompat
import java.lang.ref.WeakReference

internal data class NativeSurfaceSnapshot(val activity: Activity, val webView: WebView)

/** Window state belongs to the current Activity, not a once-loaded Tauri plugin. */
object AndroidNativeSurface {
    private var activity = WeakReference<Activity>(null)
    private var webView = WeakReference<WebView>(null)
    private var appearance: String? = null

    fun bind(owner: Activity, view: WebView) {
        activity = WeakReference(owner)
        webView = WeakReference(view)
        reapply(owner)
    }

    fun clear(owner: Activity) {
        if (activity.get() !== owner) return
        activity.clear()
        webView.clear()
    }

    internal fun snapshot(): NativeSurfaceSnapshot {
        val owner = activity.get()?.takeUnless { it.isDestroyed }
            ?: error("The app window is not available")
        val view = webView.get() ?: error("The app view is not available")
        return NativeSurfaceSnapshot(owner, view)
    }

    fun setAppearance(value: String) {
        require(value in setOf("dark", "light", "system")) { "unknown appearance $value" }
        val owner = activity.get()?.takeUnless { it.isDestroyed }
            ?: error("The app window is not available")
        apply(owner, value)
        appearance = value
    }

    fun reapply(owner: Activity) {
        if (activity.get() === owner) appearance?.let { apply(owner, it) }
    }

    @Suppress("DEPRECATION")
    private fun apply(owner: Activity, value: String) {
        val dark = when (value) {
            "dark" -> true
            "light" -> false
            else -> (owner.resources.configuration.uiMode and
                Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES
        }
        WindowCompat.getInsetsController(owner.window, owner.window.decorView).apply {
            isAppearanceLightStatusBars = !dark
            isAppearanceLightNavigationBars = !dark
        }
        val chrome = Color.parseColor(if (dark) "#0A0805" else "#E6DCC7")
        owner.window.apply {
            statusBarColor = chrome
            navigationBarColor = chrome
            decorView.setBackgroundColor(chrome)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                isStatusBarContrastEnforced = false
                isNavigationBarContrastEnforced = false
            }
        }
    }
}
