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
                val body = CharArray(contentLength)
                if (contentLength > 0) reader.read(body)
                requestText.set(headers.joinToString("\n") + "\n\n" + String(body))
                val gif = "GIF89a uat".toByteArray()
                val responseHeaders = buildString {
                    append("HTTP/1.1 200 OK\r\n")
                    append("Content-Type: image/gif\r\n")
                    append("Content-Length: ${gif.size}\r\n")
                    append("Connection: close\r\n\r\n")
                }.toByteArray()
                socket.getOutputStream().use { output ->
                    output.write(responseHeaders)
                    output.write(gif)
                    output.flush()
                }
            }
        }.apply { start() }

        try {
            val chooser = media.prepareAnimationShare(
                "http://127.0.0.1:${server.localPort}/api/gallery/export/clip.mp4",
                "uat-secret",
                "{\"format\":\"gif\"}",
                "clip.gif",
                "uat-animation-${System.nanoTime()}",
            )
            responder.join(5_000)

            assertEquals(Intent.ACTION_CHOOSER, chooser.action)
            val send = chooser.getParcelableExtra(Intent.EXTRA_INTENT, Intent::class.java)!!
            assertEquals(Intent.ACTION_SEND, send.action)
            assertEquals("image/gif", send.type)
            assertTrue(send.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION != 0)
            val uri = send.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)!!
            val shared = context.contentResolver.openInputStream(uri)!!.use { it.readBytes() }
            assertArrayEquals("GIF89a uat".toByteArray(), shared)
            assertTrue(requestText.get().contains("x-api-key: uat-secret", ignoreCase = true))
            assertTrue(requestText.get().endsWith("{\"format\":\"gif\"}"))
        } finally {
            server.close()
        }
    }

    companion object {
        // Opaque 1x1 PNG. The native bridge retains the original bytes.
        private const val PNG_BASE64 =
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    }
}
