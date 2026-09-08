package com.utensils.mold

import android.R
import android.graphics.Color
import android.content.res.Configuration
import android.os.Bundle
import android.webkit.WebView
import com.utensils.mold.mobile_native.AndroidOverlayBack
import com.utensils.mold.mobile_native.AndroidNativeSurface
import com.utensils.mold.mobile_native.applyAndroidTextScale
import androidx.activity.enableEdgeToEdge
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : TauriActivity() {
  private var currentWebView: WebView? = null

  override fun onWebViewCreate(webView: WebView) {
    super.onWebViewCreate(webView)
    currentWebView = webView
    AndroidNativeSurface.bind(this, webView)
    applyAndroidTextScale(webView, resources.configuration.fontScale)
    AndroidOverlayBack(this, webView)
  }

  override fun onConfigurationChanged(newConfig: Configuration) {
    super.onConfigurationChanged(newConfig)
    currentWebView?.let { applyAndroidTextScale(it, newConfig.fontScale) }
    AndroidNativeSurface.reapply(this)
  }

  override fun onDestroy() {
    AndroidNativeSurface.clear(this)
    currentWebView = null
    super.onDestroy()
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
    // Avoid a light flash before the WebView applies the persisted appearance.
    window.decorView.setBackgroundColor(Color.parseColor("#0A0805"))
    val content = findViewById<android.view.View>(R.id.content)
    ViewCompat.setOnApplyWindowInsetsListener(content) { view, insets ->
      val safeChrome = insets.getInsets(
        WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout(),
      )
      val keyboard = insets.getInsets(WindowInsetsCompat.Type.ime())
      view.setPadding(
        safeChrome.left,
        safeChrome.top,
        safeChrome.right,
        maxOf(safeChrome.bottom, keyboard.bottom),
      )
      // This container already applies system and IME insets. Do not pass them
      // to WebView, which would expose them again through CSS safe-area values.
      WindowInsetsCompat.CONSUMED
    }
    ViewCompat.requestApplyInsets(content)
    AndroidNativeSurface.reapply(this)
  }
}
