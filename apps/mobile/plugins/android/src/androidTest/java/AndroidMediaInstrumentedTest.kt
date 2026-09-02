package com.utensils.mold.mobile_native

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Base64
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.ServerSocket
import java.util.concurrent.atomic.AtomicReference

class AndroidMediaInstrumentedTest {
    private val context: Context = ApplicationProvider.getApplicationContext()
    private val media = AndroidMedia(context)

    @Test
    fun requestsLegacyMediaPermissionOnlyBeforeScopedStorage() {
        assertTrue(needsLegacyMediaWritePermission(28))
        assertFalse(needsLegacyMediaWritePermission(29))
    }

    @Test
    fun copiesImageAsReadableContentUri() {
        val uri = media.copyImage(PNG_BASE64)

        assertNotNull(uri)
        assertEquals("content", uri.scheme)
        val copied = context.contentResolver.openInputStream(uri)!!.use { it.readBytes() }
        assertArrayEquals(Base64.decode(PNG_BASE64, Base64.DEFAULT), copied)
    }

    @Test
    fun savesImageThroughMediaStoreWithoutBroadPhotoReadAccess() {
        val uri = media.saveImage(PNG_BASE64)
        try {
            assertEquals("content", uri.scheme)
            val saved = context.contentResolver.openInputStream(uri)!!.use { it.readBytes() }
            assertArrayEquals(Base64.decode(PNG_BASE64, Base64.DEFAULT), saved)
        } finally {
            context.contentResolver.delete(uri, null, null)
        }
    }

    @Test
    fun preparesAuthenticatedAnimationForTheAndroidShareSheet() {
        val host = startExportHost("GIF89a uat".toByteArray(), "image/gif")
        try {
            val chooser = media.prepareExportShare(
                "http://127.0.0.1:${host.server.localPort}/api/gallery/export/clip.mp4",
                "uat-secret",
                "{\"format\":\"gif\"}",
                "clip.gif",
                "image/gif",
                "uat-animation-${System.nanoTime()}",
            )
            host.responder.join(5_000)

            assertEquals(Intent.ACTION_CHOOSER, chooser.action)
            val send = chooser.getParcelableExtra(Intent.EXTRA_INTENT, Intent::class.java)!!
            assertEquals(Intent.ACTION_SEND, send.action)
            assertEquals("image/gif", send.type)
            assertTrue(send.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION != 0)
            val uri = send.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)!!
            val shared = context.contentResolver.openInputStream(uri)!!.use { it.readBytes() }
            assertArrayEquals("GIF89a uat".toByteArray(), shared)
            assertTrue(host.requestText.get().contains("x-api-key: uat-secret", ignoreCase = true))
            assertTrue(host.requestText.get().endsWith("{\"format\":\"gif\"}"))
        } finally {
            host.server.close()
        }
    }

    /**
     * A mesh geometry transcode takes the same native route a turntable does:
     * the chooser advertises the type the app resolved, and the glTF bytes
     * reach it unchanged.
     */
    @Test
    fun preparesMeshGeometryForTheAndroidShareSheet() {
        val glb = "glTF".toByteArray() + byteArrayOf(2, 0, 0, 0)
        val host = startExportHost(glb, "model/gltf-binary")
        try {
            val chooser = media.prepareExportShare(
                "http://127.0.0.1:${host.server.localPort}/api/gallery/export/armchair.glb",
                null,
                "{\"format\":\"glb\"}",
                "armchair.glb",
                "model/gltf-binary",
                "uat-mesh-${System.nanoTime()}",
            )
            host.responder.join(5_000)

            val send = chooser.getParcelableExtra(Intent.EXTRA_INTENT, Intent::class.java)!!
            assertEquals("model/gltf-binary", send.type)
            val uri = send.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)!!
            val shared = context.contentResolver.openInputStream(uri)!!.use { it.readBytes() }
            assertArrayEquals(glb, shared)
            assertTrue(host.requestText.get().endsWith("{\"format\":\"glb\"}"))
        } finally {
            host.server.close()
        }
    }

    private class ExportHost(
        val server: ServerSocket,
        val responder: Thread,
        val requestText: AtomicReference<String>,
    )

    /** A one-shot host that answers the export POST with exactly `body`. */
    private fun startExportHost(body: ByteArray, contentType: String): ExportHost {
        val server = ServerSocket(0)
        val requestText = AtomicReference("")
        val responder = Thread {
            server.accept().use { socket ->
                val reader = socket.getInputStream().bufferedReader()
                val headers = mutableListOf<String>()
                var contentLength = 0
                while (true) {
                    val line = reader.readLine() ?: break
                    if (line.isEmpty()) break
                    headers += line
                    if (line.startsWith("Content-Length:", ignoreCase = true)) {
                        contentLength = line.substringAfter(':').trim().toInt()
                    }
                }
                val requestBody = CharArray(contentLength)
                if (contentLength > 0) reader.read(requestBody)
                requestText.set(headers.joinToString("\n") + "\n\n" + String(requestBody))
                val responseHeaders = buildString {
                    append("HTTP/1.1 200 OK\r\n")
                    append("Content-Type: $contentType\r\n")
                    append("Content-Length: ${body.size}\r\n")
                    append("Connection: close\r\n\r\n")
                }.toByteArray()
                socket.getOutputStream().use { output ->
                    output.write(responseHeaders)
                    output.write(body)
                    output.flush()
                }
            }
        }.apply { start() }
        return ExportHost(server, responder, requestText)
    }

    companion object {
        // Opaque 1x1 PNG. The native bridge retains the original bytes.
        private const val PNG_BASE64 =
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    }
}
