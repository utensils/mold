package com.utensils.mold.mobile_native

import android.os.Handler
import android.os.Looper
import android.webkit.WebView
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import java.lang.ref.WeakReference

/** Ask the live overlay stack before relying on WebView's asynchronous history mirror. */
class AndroidOverlayBack(
    private val activity: ComponentActivity,
    webView: WebView,
) : DefaultLifecycleObserver {
    private val webView = WeakReference(webView)
    private val handler = Handler(Looper.getMainLooper())
    private var pending: Runnable? = null
    private var destroyed = false
    private val callback = object : OnBackPressedCallback(true) {
        override fun handleOnBackPressed() {
            if (pending != null || destroyed) return
            val view = this@AndroidOverlayBack.webView.get()
            if (view == null) {
                delegate()
                return
            }
            // Expired work must not close a later surface. On timeout, release
            // the attempt without exiting: the renderer may have consumed Back
            // just before its response was delayed on the way to the UI thread.
            val deadline = System.currentTimeMillis() + 1000
            val timeout = Runnable {
                pending = null
            }
            pending = timeout
            handler.postDelayed(timeout, 1000)
            try {
                view.evaluateJavascript(
                    "Date.now() < $deadline && !window.dispatchEvent(new Event('mold:native-back', {cancelable:true}))",
                ) { consumed ->
                    if (pending !== timeout || destroyed) return@evaluateJavascript
                    handler.removeCallbacks(timeout)
                    pending = null
                    if (consumed != "true") delegate()
                }
            } catch (_: RuntimeException) {
                handler.removeCallbacks(timeout)
                pending = null
                delegate()
            }
        }
    }

    init {
        activity.lifecycle.addObserver(this)
        activity.onBackPressedDispatcher.addCallback(activity, callback)
    }

    private fun delegate() {
        callback.isEnabled = false
        try {
            activity.onBackPressedDispatcher.onBackPressed()
        } finally {
            if (!destroyed) callback.isEnabled = true
        }
    }

    override fun onDestroy(owner: LifecycleOwner) {
        destroyed = true
        pending?.let(handler::removeCallbacks)
        pending = null
        callback.remove()
        webView.clear()
        activity.lifecycle.removeObserver(this)
    }
}
