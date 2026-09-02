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
            // The chooser shows the file's own name, so it is staged under it.
            assertEquals("clip.gif", uri.lastPathSegment)
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
            assertEquals("armchair.glb", uri.lastPathSegment)
            val shared = context.contentResolver.openInputStream(uri)!!.use { it.readBytes() }
            assertArrayEquals(glb, shared)
            assertTrue(host.requestText.get().endsWith("{\"format\":\"glb\"}"))
        } finally {
            host.server.close()
        }
    }

    /**
     * "Save to Mold folder" is the share's twin: the same one-shot host, the
     * same byte check, but the glTF lands under Downloads/Mold as a readable
     * MediaStore item and the label names that place for the toast.
     */
    @Test
    fun filesMeshGeometryUnderTheDownloadsMoldFolder() {
        val glb = "glTF".toByteArray() + byteArrayOf(2, 0, 0, 0)
        val host = startExportHost(glb, "model/gltf-binary")
        val name = "armchair-${System.nanoTime()}.glb"
        var location: Uri? = null
        try {
            val saved = media.saveExportToMoldFolder(
                "http://127.0.0.1:${host.server.localPort}/api/gallery/export/armchair.glb",
                null,
                "{\"format\":\"glb\"}",
                name,
                "model/gltf-binary",
                "uat-mold-folder-${System.nanoTime()}",
            )
            host.responder.join(5_000)
            location = Uri.parse(saved.location)

            assertEquals(name, saved.filename)
            assertEquals("Downloads/Mold/$name", saved.label)
            val written = context.contentResolver.openInputStream(location)!!.use { it.readBytes() }
            assertArrayEquals(glb, written)
            assertTrue(host.requestText.get().endsWith("{\"format\":\"glb\"}"))
        } finally {
            host.server.close()
            location?.let { context.contentResolver.delete(it, null, null) }
        }
    }

    /**
     * A second save of the same print is numbered the way the iPhone numbers
     * it — `name (2).ext`, extension kept — rather than left to MediaStore,
     * which does not know the geometry media types and would answer
     * `name.stl (1)`.
     */
    @Test
    fun numbersASecondMoldFolderSaveBeforeTheExtension() {
        val stl = "solid armchair\nendsolid armchair\n".toByteArray()
        val stem = "armchair-${System.nanoTime()}"
        val locations = mutableListOf<Uri>()
        try {
            repeat(2) { attempt ->
                val host = startExportHost(stl, "model/stl")
                try {
                    val saved = media.saveExportToMoldFolder(
                        "http://127.0.0.1:${host.server.localPort}/api/gallery/export/armchair.glb",
                        null,
                        "{\"format\":\"stl\"}",
                        "$stem.stl",
                        "model/stl",
                        "uat-mold-folder-$stem-$attempt",
                    )
                    host.responder.join(5_000)
                    locations += Uri.parse(saved.location)
                    val expected = if (attempt == 0) "$stem.stl" else "$stem (2).stl"
                    assertTrue(saved.filename.endsWith(".stl"))
                    assertEquals(expected, saved.filename)
                    assertEquals("Downloads/Mold/$expected", saved.label)
                } finally {
                    host.server.close()
                }
            }
        } finally {
            locations.forEach { context.contentResolver.delete(it, null, null) }
        }
    }

    /** A second export of the same print gets a numbered name, never an overwrite. */
    @Test
    fun numbersAMoldFolderCollisionInsteadOfOverwriting() {
        val directory = java.io.File(context.cacheDir, "mold-folder-test-${System.nanoTime()}").apply { mkdirs() }
        try {
            assertEquals("chair (2).stl", AndroidMedia.uniqueName("chair.stl") { it == "chair.stl" })
            assertEquals("chair.stl", AndroidMedia.uniqueDestination(directory, "chair.stl").name)
            java.io.File(directory, "chair.stl").writeText("solid chair")
            assertEquals("chair (2).stl", AndroidMedia.uniqueDestination(directory, "chair.stl").name)
            java.io.File(directory, "chair (2).stl").writeText("solid chair")
            assertEquals("chair (3).stl", AndroidMedia.uniqueDestination(directory, "chair.stl").name)
            java.io.File(directory, "armchair.v2.glb").writeText("glTF")
            assertEquals("armchair.v2 (2).glb", AndroidMedia.uniqueDestination(directory, "armchair.v2.glb").name)
        } finally {
            directory.deleteRecursively()
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
