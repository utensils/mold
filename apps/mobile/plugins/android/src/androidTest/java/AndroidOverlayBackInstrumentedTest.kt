package com.utensils.mold.mobile_native

import android.content.Intent
import android.os.Bundle
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Real dispatcher -> WebView JS -> canceled overlay Back, without a server. */
class AndroidOverlayBackInstrumentedTest {
    @Test
    fun dispatcherClosesAnOverlayWithoutDelegating() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val intent = Intent(instrumentation.context, OverlayBackTestActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        ActivityScenario.launch<OverlayBackTestActivity>(intent).use { scenario ->
            var activity: OverlayBackTestActivity? = null
            scenario.onActivity { activity = it }
            assertTrue("test page loaded", activity!!.ready.await(10, TimeUnit.SECONDS))
            scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
            val answered = CountDownLatch(1)
            var closed: String? = null
            // A posted evaluation follows the Back evaluation on the renderer queue.
            scenario.onActivity {
                it.view.evaluateJavascript("window.closedByBack") { value ->
                    closed = value
                    answered.countDown()
                }
            }
            assertTrue("renderer answered", answered.await(5, TimeUnit.SECONDS))
            assertEquals("one overlay was closed", "1", closed)
            scenario.onActivity { assertEquals("Back was consumed", 0, it.delegated) }
        }
    }
}

class OverlayBackTestActivity : ComponentActivity() {
    lateinit var view: WebView
    val ready = CountDownLatch(1)
    var delegated = 0
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        view = WebView(this)
        view.settings.javaScriptEnabled = true
        setContentView(view)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() { delegated++ }
        })
        AndroidOverlayBack(this, view)
        view.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView, url: String) { ready.countDown() }
        }
        view.loadData("""<script>window.closedByBack=0;window.addEventListener('mold:native-back',e=>{window.closedByBack++;e.preventDefault()})</script>""", "text/html", "UTF-8")
    }
    override fun onDestroy() {
        super.onDestroy()
        view.destroy()
    }
}
