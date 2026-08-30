package com.utensils.mold.mobile_native

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.Drawable
import android.view.ViewGroup
import android.webkit.WebView
import android.widget.FrameLayout
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import com.google.mlkit.vision.barcode.BarcodeScanner
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage

internal data class ActivePairingScan<T>(val id: Long, val value: T)

/** Owns exactly one pending invocation and invalidates late CameraX callbacks. */
internal class PairingScanSession<T> {
    private var nextId = 0L
    private var active: ActivePairingScan<T>? = null

    @Synchronized
    fun begin(value: T): Long {
        check(active == null) { "a pairing scan is already active" }
        nextId += 1
        active = ActivePairingScan(nextId, value)
        return nextId
    }

    @Synchronized
    fun isActive(id: Long): Boolean = active?.id == id

    @Synchronized
    fun complete(id: Long): T? {
        val current = active?.takeIf { it.id == id } ?: return null
        active = null
        return current.value
    }

    @Synchronized
    fun cancel(): ActivePairingScan<T>? = active.also {
        active = null
        nextId += 1
    }
}

/** CameraX + bundled ML Kit scanner with deterministic cancellation semantics. */
internal class AndroidPairingScanner(
    private val activity: Activity,
    private val webView: WebView,
) {
    private val session = PairingScanSession<Invoke>()
    private var previewView: PreviewView? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var decoder: BarcodeScanner? = null
    private var webViewBackground: Drawable? = null

    fun scan(invoke: Invoke) {
        if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            invoke.reject("Camera access is required to scan a pairing code.")
            return
        }
        val scanId = try {
            session.begin(invoke)
        } catch (error: IllegalStateException) {
            invoke.reject(error.message ?: "a pairing scan is already active")
            return
        }
        activity.runOnUiThread { attachCamera(scanId) }
    }

    fun cancel(cancelInvoke: Invoke) {
        val pending = session.cancel()
        activity.runOnUiThread {
            cleanupCamera()
            pending?.value?.reject("cancelled")
            cancelInvoke.resolve()
        }
    }

    private fun attachCamera(scanId: Long) {
        if (!session.isActive(scanId)) return
        try {
            val parent = webView.parent as? ViewGroup
                ?: error("Android WebView has no camera overlay parent")
            previewView = PreviewView(activity).apply {
                layoutParams = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                )
            }
            parent.addView(previewView)
            webViewBackground = webView.background
            webView.setBackgroundColor(Color.TRANSPARENT)
            webView.bringToFront()

            decoder = BarcodeScanning.getClient(
                BarcodeScannerOptions.Builder()
                    .setBarcodeFormats(Barcode.FORMAT_QR_CODE)
                    .build(),
            )
            val providerFuture = ProcessCameraProvider.getInstance(activity)
            providerFuture.addListener({
                if (!session.isActive(scanId)) return@addListener
                try {
                    bindCamera(scanId, providerFuture.get())
                } catch (error: Exception) {
                    fail(scanId, "could not start the Android camera: ${error.message ?: error.javaClass.simpleName}")
                }
            }, ContextCompat.getMainExecutor(activity))
        } catch (error: Exception) {
            fail(scanId, "could not open the Android pairing scanner: ${error.message ?: error.javaClass.simpleName}")
        }
    }

    private fun bindCamera(scanId: Long, provider: ProcessCameraProvider) {
        if (!session.isActive(scanId)) return
        val preview = Preview.Builder().build().apply {
            setSurfaceProvider(previewView?.surfaceProvider)
        }
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .apply {
                setAnalyzer(ContextCompat.getMainExecutor(activity)) { image ->
                    analyze(scanId, image)
                }
            }
        provider.unbindAll()
        provider.bindToLifecycle(
            activity as LifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            analysis,
        )
        cameraProvider = provider
    }

    @android.annotation.SuppressLint("UnsafeOptInUsageError")
    private fun analyze(scanId: Long, image: ImageProxy) {
        val mediaImage = image.image
        if (mediaImage == null || !session.isActive(scanId)) {
            image.close()
            return
        }
        decoder?.process(InputImage.fromMediaImage(mediaImage, image.imageInfo.rotationDegrees))
            ?.addOnSuccessListener { barcodes ->
                val content = barcodes.firstOrNull()?.rawValue?.takeIf { it.isNotBlank() }
                if (content != null) finish(scanId, content)
            }
            ?.addOnFailureListener { error ->
                fail(scanId, "could not read the pairing code: ${error.message ?: error.javaClass.simpleName}")
            }
            ?.addOnCompleteListener { image.close() }
            ?: image.close()
    }

    private fun finish(scanId: Long, content: String) {
        val invoke = session.complete(scanId) ?: return
        activity.runOnUiThread {
            cleanupCamera()
            invoke.resolve(JSObject().apply { put("content", content) })
        }
    }

    private fun fail(scanId: Long, message: String) {
        val invoke = session.complete(scanId) ?: return
        activity.runOnUiThread {
            cleanupCamera()
            invoke.reject(message)
        }
    }

    private fun cleanupCamera() {
        cameraProvider?.unbindAll()
        cameraProvider = null
        decoder?.close()
        decoder = null
        val parent = webView.parent as? ViewGroup
        previewView?.let { parent?.removeView(it) }
        previewView = null
        webView.background = webViewBackground
        webViewBackground = null
    }
}
