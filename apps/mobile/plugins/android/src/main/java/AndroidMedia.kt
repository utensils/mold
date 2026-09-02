package com.utensils.mold.mobile_native

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap

internal class AndroidMedia(context: Context) {
    private val context = context.applicationContext

    fun copyImage(dataB64: String): Uri {
        val clip = prepareImageClip(dataB64)
        copyPreparedImage(clip)
        return clip.uri
    }

    fun prepareImageClip(dataB64: String): PreparedImageClip {
        val image = decodeImage(dataB64)
        val directory = File(context.cacheDir, "shared").apply { mkdirs() }
        prune(directory)
        val file = File(directory, "mold-copy-${System.currentTimeMillis()}.${image.extension}")
        file.writeBytes(image.bytes)
        return PreparedImageClip(shareUri(file))
    }

    fun copyPreparedImage(clip: PreparedImageClip) {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newUri(context.contentResolver, "Mold image", clip.uri))
    }

    fun saveImage(dataB64: String): Uri {
        val image = decodeImage(dataB64)
        val name = "Mold-${System.currentTimeMillis()}.${image.extension}"
        return saveBytes(
            image.bytes,
            name,
            image.mimeType,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            Environment.DIRECTORY_PICTURES,
        )
    }

    fun saveVideo(url: String): Uri {
        requireModernStoragePermission()
        val connection = openConnection(url, "GET")
        try {
            requireSuccessful(connection, "download the video")
            val mimeType = connection.contentType?.substringBefore(';')
                ?.takeIf { it.startsWith("video/") } ?: "video/mp4"
            val extension = when (mimeType) {
                "video/webm" -> "webm"
                "video/quicktime" -> "mov"
                else -> "mp4"
            }
            return writeMediaStream(
                connection.inputStream,
                "Mold-${System.currentTimeMillis()}.$extension",
                mimeType,
                MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                Environment.DIRECTORY_MOVIES,
            )
        } finally {
            connection.disconnect()
        }
    }

    /**
     * One gallery export handed to the Android chooser: a turntable or clip
     * animation, or a geometry transcode of a stored mesh. The media type is
     * the app's answer, resolved once from the Rust share allowlist, so the
     * chooser and the byte check below can never disagree about a container.
     */
    fun prepareExportShare(
        url: String,
        apiKey: String?,
        requestJson: String,
        filename: String,
        mimeType: String,
        reuseKey: String,
    ): Intent {
        val file = stagedExport(url, apiKey, requestJson, filename, mimeType, reuseKey)
        val uri = shareUri(file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = mimeType
            putExtra(Intent.EXTRA_STREAM, uri)
            clipData = ClipData.newRawUri("Mold export", uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(send, "Share Mold export")
    }

    /**
     * The other half of the export pair: the same download, cache and byte
     * check as [prepareExportShare], but the file is filed under the public
     * `Downloads/Mold` folder instead of handed to a chooser. API 29+ goes
     * through MediaStore's Downloads collection (no storage permission);
     * earlier releases write the
     * public Downloads directory directly, with the app's own external files
     * directory as the fallback when that volume is unavailable.
     */
    fun saveExportToMoldFolder(
        url: String,
        apiKey: String?,
        requestJson: String,
        filename: String,
        mimeType: String,
        reuseKey: String,
    ): SavedExport {
        val file = stagedExport(url, apiKey, requestJson, filename, mimeType, reuseKey)
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            saveExportThroughMediaStore(file, file.name, mimeType)
        } else {
            saveExportToPublicDownloads(file, file.name, mimeType)
        }
    }

    /**
     * The staged, validated file for one export: what a share the user
     * backed out of left under [reuseKey], or a fresh download. Both doors —
     * the chooser and the Mold folder — start here, so they validate
     * identically.
     */
    private fun stagedExport(
        url: String,
        apiKey: String?,
        requestJson: String,
        filename: String,
        mimeType: String,
        reuseKey: String,
    ): File {
        val safeName = File(filename).name
        require(safeName == filename && safeName.isNotBlank()) { "invalid export filename" }
        require(mimeType.isNotBlank()) { "the export does not name a media type" }
        val cached = exportCache[reuseKey]?.takeIf { it.isFile }
        val file = cached ?: downloadExport(url, apiKey, requestJson, safeName).also {
            exportCache[reuseKey] = it
        }
        requireMatchingExport(file)
        return file
    }

    private fun saveExportThroughMediaStore(file: File, name: String, mimeType: String): SavedExport {
        val resolver = context.contentResolver
        // Number a collision ourselves, before the extension, the way the
        // iPhone does. Left to MediaStore the geometry types it does not know
        // come back as `chair.stl (1)` — the extension lost behind the number.
        val chosenName = uniqueName(name) { candidate -> moldFolderHasDownload(candidate) }
        val uri = file.inputStream().use { input ->
            writeMediaStream(
                input,
                chosenName,
                mimeType,
                MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                Environment.DIRECTORY_DOWNLOADS,
            )
        }
        // A file this app cannot see (another app's) can still collide, and
        // MediaStore then renames; the name the toast shows is read back.
        val displayName = resolver.query(
            uri,
            arrayOf(MediaStore.MediaColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        } ?: chosenName
        return SavedExport(displayName, uri.toString(), "$MOLD_FOLDER_LABEL/$displayName")
    }

    /** Whether this app already filed [name] under Downloads/Mold through MediaStore. */
    private fun moldFolderHasDownload(name: String): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return false
        val relative = "${Environment.DIRECTORY_DOWNLOADS}/$MOLD_FOLDER_NAME"
        return context.contentResolver.query(
            MediaStore.Downloads.EXTERNAL_CONTENT_URI,
            arrayOf(MediaStore.MediaColumns._ID),
            "${MediaStore.MediaColumns.DISPLAY_NAME} = ? AND ${MediaStore.MediaColumns.RELATIVE_PATH} IN (?, ?)",
            arrayOf(name, "$relative/", relative),
            null,
        )?.use { cursor -> cursor.count > 0 } ?: false
    }

    /**
     * Before scoped storage: the public `Download/Mold` directory itself,
     * behind the legacy storage permission. There is no app-private fallback
     * — a file the Downloads app cannot list is not "saved to the Mold
     * folder" — so an unavailable volume is an error, and a successful write
     * is handed to the media scanner so the Downloads app lists it at once.
     */
    @Suppress("DEPRECATION")
    private fun saveExportToPublicDownloads(file: File, name: String, mimeType: String): SavedExport {
        requireLegacyStoragePermission("Storage access is required to save into the Mold folder")
        val directory = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), MOLD_FOLDER_NAME)
        check(directory.isDirectory || directory.mkdirs()) {
            "could not create the Download/Mold folder on this phone's storage"
        }
        val destination = uniqueDestination(directory, name)
        try {
            file.inputStream().use { input -> FileOutputStream(destination).use { output -> input.copyTo(output) } }
        } catch (error: IOException) {
            // Never leave a partial file where the user will look for it.
            destination.delete()
            throw error
        }
        MediaScannerConnection.scanFile(context, arrayOf(destination.absolutePath), arrayOf(mimeType), null)
        return SavedExport(destination.name, Uri.fromFile(destination).toString(), "$MOLD_FOLDER_LABEL/${destination.name}")
    }

    /**
     * Download one export into a directory of its own under the shared
     * cache, holding the file under its REAL name: the chooser and any
     * "save" from it show the file's name, so staging as
     * `mold-export-123-chair.stl` would present that mangled name. The
     * unique part lives on the directory instead.
     */
    private fun downloadExport(
        url: String,
        apiKey: String?,
        requestJson: String,
        filename: String,
    ): File {
        val shared = File(context.cacheDir, "shared").apply { mkdirs() }
        prune(shared)
        val directory = File(shared, "$STAGED_EXPORT_PREFIX${System.nanoTime()}").apply { mkdirs() }
        val file = File(directory, filename)
        val connection = openConnection(url, "POST").apply {
            setRequestProperty("Content-Type", "application/json")
            if (!apiKey.isNullOrBlank()) setRequestProperty("x-api-key", apiKey)
            doOutput = true
            outputStream.use { it.write(requestJson.toByteArray(Charsets.UTF_8)) }
        }
        try {
            requireSuccessful(connection, "export this print")
            BufferedInputStream(connection.inputStream).use { input ->
                FileOutputStream(file).use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var total = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        total += read
                        require(total <= MAX_EXPORT_BYTES) { "the export exceeds the 2 GB Android limit" }
                        output.write(buffer, 0, read)
                    }
                    output.fd.sync()
                }
            }
            requireMatchingExport(file)
            return file
        } catch (error: Exception) {
            file.delete()
            directory.delete()
            throw error
        } finally {
            connection.disconnect()
        }
    }

    private fun saveBytes(
        bytes: ByteArray,
        name: String,
        mimeType: String,
        collection: Uri,
        directory: String,
    ): Uri {
        requireModernStoragePermission()
        return writeMediaStream(bytes.inputStream(), name, mimeType, collection, directory)
    }

    private fun writeMediaStream(
        source: java.io.InputStream,
        name: String,
        mimeType: String,
        collection: Uri,
        directory: String,
    ): Uri {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.MediaColumns.RELATIVE_PATH, "$directory/Mold")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(collection, values)
            ?: error("Android MediaStore could not create the media item")
        try {
            source.use { input ->
                resolver.openOutputStream(uri, "w")?.use { output -> input.copyTo(output) }
                    ?: error("Android MediaStore could not open the media item")
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                resolver.update(uri, ContentValues().apply {
                    put(MediaStore.MediaColumns.IS_PENDING, 0)
                }, null, null)
            }
            return uri
        } catch (error: Exception) {
            resolver.delete(uri, null, null)
            throw error
        }
    }

    private fun requireModernStoragePermission() {
        requireLegacyStoragePermission("Photos access is required on this Android version")
    }

    private fun requireLegacyStoragePermission(message: String) {
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            check(
                ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE) ==
                    PackageManager.PERMISSION_GRANTED,
            ) { message }
        }
    }

    private fun decodeImage(dataB64: String): ImageData {
        val bytes = try {
            Base64.decode(dataB64, Base64.DEFAULT)
        } catch (_: IllegalArgumentException) {
            throw IllegalArgumentException("the selected print contains invalid image data")
        }
        return when {
            bytes.size >= PNG_SIGNATURE.size && bytes.copyOfRange(0, PNG_SIGNATURE.size).contentEquals(PNG_SIGNATURE) ->
                ImageData(bytes, "image/png", "png")
            bytes.size >= 3 && bytes[0] == 0xff.toByte() && bytes[1] == 0xd8.toByte() && bytes[2] == 0xff.toByte() ->
                ImageData(bytes, "image/jpeg", "jpg")
            else -> throw IllegalArgumentException("the selected print is not a PNG or JPEG image")
        }
    }

    /**
     * Whether a download really is the container its filename claims. Binary
     * STL carries no signature at all — its 84-byte header states the facet
     * count, and the file length is the only check — so the probe reads the
     * whole header rather than the 12 bytes an animation needs.
     */
    private fun requireMatchingExport(file: File) {
        val probe = ByteArray(EXPORT_HEADER_PROBE_BYTES)
        val count = file.inputStream().use { it.read(probe) }.coerceAtLeast(0)
        val head = probe.copyOfRange(0, count)
        val valid = when (file.extension.lowercase()) {
            "gif" -> head.startsWith("GIF87a".toByteArray()) || head.startsWith("GIF89a".toByteArray())
            // APNG carries the ordinary PNG signature.
            "png" -> head.startsWith(PNG_SIGNATURE)
            "webp" -> count >= 12 && head.copyOfRange(0, 4).contentEquals("RIFF".toByteArray()) &&
                head.copyOfRange(8, 12).contentEquals("WEBP".toByteArray())
            "glb" -> head.startsWith("glTF".toByteArray())
            "obj" -> firstContentLine(head)?.let { line ->
                OBJ_LINE_PREFIXES.any { prefix -> line.startsWith(prefix) }
            } ?: false
            "stl" -> head.startsWith("solid".toByteArray()) || binaryStlCovers(head, file.length())
            "ply" -> head.startsWith("ply\n".toByteArray()) || head.startsWith("ply\r\n".toByteArray())
            else -> false
        }
        require(valid) { "the exported file does not match the format its filename claims" }
    }

    /**
     * The first line of a text export that carries content. Latin-1 keeps
     * every byte addressable, and a probe that stops mid-line still answers:
     * every prefix that identifies an OBJ is far shorter than the probe.
     */
    private fun firstContentLine(head: ByteArray): String? =
        String(head, Charsets.ISO_8859_1).split('\n').map { it.trim() }.firstOrNull { it.isNotEmpty() }

    /** Binary STL: 80-byte comment, little-endian facet count, 50 bytes each. */
    private fun binaryStlCovers(head: ByteArray, totalBytes: Long): Boolean {
        if (head.size < 84) return false
        val facets = ByteBuffer.wrap(head, 80, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL
        return 84L + 50L * facets == totalBytes
    }

    private fun ByteArray.startsWith(prefix: ByteArray): Boolean =
        size >= prefix.size && copyOfRange(0, prefix.size).contentEquals(prefix)

    private fun openConnection(url: String, method: String): HttpURLConnection =
        (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 30_000
            readTimeout = 600_000
            instanceFollowRedirects = true
        }

    private fun requireSuccessful(connection: HttpURLConnection, action: String) {
        val code = connection.responseCode
        check(code in 200..299) { "could not $action: host returned HTTP $code" }
    }

    private fun shareUri(file: File): Uri = FileProvider.getUriForFile(
        context,
        "${context.packageName}.mold-mobile-native-fileprovider",
        file,
    )

    /**
     * Drop shared-cache entries older than a day: the per-export staging
     * directories, the flat files older builds staged, and copied images.
     */
    private fun prune(directory: File) {
        val cutoff = System.currentTimeMillis() - CACHE_MAX_AGE_MS
        directory.listFiles()?.filter { it.lastModified() < cutoff }?.forEach { stale ->
            exportCache.entries.removeIf { it.value == stale || it.value.parentFile == stale }
            if (stale.isDirectory) stale.deleteRecursively() else stale.delete()
        }
    }

    internal data class PreparedImageClip(val uri: Uri)

    /**
     * Where a Mold-folder save landed: the final name (numbered on a
     * collision), the `content://` or `file://` location, and the
     * `Downloads/Mold/<name>` label the toast shows.
     */
    internal data class SavedExport(val filename: String, val location: String, val label: String)

    private data class ImageData(val bytes: ByteArray, val mimeType: String, val extension: String)

    companion object {
        private val PNG_SIGNATURE = byteArrayOf(
            0x89.toByte(), 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
        )
        private const val MAX_EXPORT_BYTES = 2L * 1024 * 1024 * 1024
        private const val CACHE_MAX_AGE_MS = 24L * 60 * 60 * 1000
        /** Enough to reach a binary STL's facet count at offset 80. */
        private const val EXPORT_HEADER_PROBE_BYTES = 84
        /** The statements a Wavefront OBJ may legally open with. */
        private val OBJ_LINE_PREFIXES = listOf("#", "v ", "vn ", "vt ", "f ", "o ", "g ", "mtllib")
        private val exportCache = ConcurrentHashMap<String, File>()
        /** The prefix of every per-export staging directory under the shared cache. */
        private const val STAGED_EXPORT_PREFIX = "mold-export-"
        /**
         * The on-device folder every "Save to Mold folder" export lands in:
         * `Download/Mold` on external storage (the Files app titles the
         * parent "Downloads").
         */
        internal const val MOLD_FOLDER_NAME = "Mold"
        /** How the toast names that folder: MediaStore's Downloads plus ours. */
        internal const val MOLD_FOLDER_LABEL = "Downloads/$MOLD_FOLDER_NAME"

        /**
         * The first free name for [filename] inside [directory]: the name
         * itself, then `name (2).ext`, `name (3).ext`, … so a second export
         * of the same print never overwrites the first. Mirrors the iPhone
         * shell's numbering. The MediaStore path applies the same rule through
         * [uniqueName], because MediaStore's own numbering does not know the
         * geometry media types and would answer `chair.stl (1)`.
         */
        internal fun uniqueDestination(directory: File, filename: String): File =
            File(directory, uniqueName(filename) { File(directory, it).exists() })

        /** [uniqueDestination] over any notion of "taken": a directory, or MediaStore. */
        internal fun uniqueName(filename: String, taken: (String) -> Boolean): String {
            if (!taken(filename)) return filename
            val stem = filename.substringBeforeLast('.', filename)
            val extension = if (filename.contains('.')) filename.substringAfterLast('.') else ""
            return generateSequence(2) { it + 1 }
                .map { number -> if (extension.isEmpty()) "$stem ($number)" else "$stem ($number).$extension" }
                .first { !taken(it) }
        }
    }
}
