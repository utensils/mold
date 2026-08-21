package com.utensils.mold

import android.R
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : TauriActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
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
      insets
    }
    ViewCompat.requestApplyInsets(content)
  }
}
