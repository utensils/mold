package com.utensils.mold.mobile_native

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Base64
import androidx.core.content.FileProvider
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream

internal const val IDENTITY_PHOTO_MAX_BYTES = 16L * 1024 * 1024

internal fun readIdentityPhotoBytes(input: InputStream, expectedSize: Long): ByteArray {
    val output = ByteArrayOutputStream(expectedSize.toInt())
    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
    var total = 0L
    while (true) {
        val count = input.read(buffer)
        if (count < 0) break
        total += count
        require(total <= IDENTITY_PHOTO_MAX_BYTES) {
            "Identity photo must be 16 MiB or smaller."
        }
        output.write(buffer, 0, count)
    }
    require(total == expectedSize) {
        "Identity photo changed while it was being read. Choose it again."
    }
    return output.toByteArray()
}

internal data class AndroidIdentityPhotoResult(
    val filename: String,
    val mimeType: String,
    val sizeBytes: Long,
    val dataB64: String,
)

/** Android-only identity acquisition. Product policy remains in the shared Studio module. */
internal class AndroidIdentityPhoto(private val context: Context) {
    fun libraryIntent(sdkInt: Int = Build.VERSION.SDK_INT): Intent =
        if (sdkInt >= Build.VERSION_CODES.TIRAMISU) {
            Intent(MediaStore.ACTION_PICK_IMAGES).apply { type = "image/*" }
        } else {
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "image/*"
            }
        }

    fun createCameraTarget(): CameraTarget {
        val directory = File(context.cacheDir, "identity").apply { mkdirs() }
        directory.listFiles()?.filter { it.isFile }?.forEach { it.delete() }
        val file = File(directory, "identity-${System.currentTimeMillis()}.jpg")
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.mold-mobile-native-fileprovider",
            file,
        )
        return CameraTarget(file, uri)
    }

    fun cameraIntent(target: CameraTarget): Intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
        putExtra(MediaStore.EXTRA_OUTPUT, target.uri)
        clipData = ClipData.newRawUri("Identity photo", target.uri)
        addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }

    /** Bounds decoded pixels before reading oversized provider data into memory. */
    fun readPicked(uri: Uri, cameraFile: File? = null): AndroidIdentityPhotoResult {
        val metadata = if (cameraFile != null) {
            IdentityMetadata(cameraFile.name, "image/jpeg", cameraFile.length())
        } else {
            queryMetadata(uri)
        }
        require(metadata.mimeType == "image/png" || metadata.mimeType == "image/jpeg") {
            "Identity photo must be a PNG or JPEG image."
        }

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
            Unit // Bounds-only decoding intentionally returns a null Bitmap.
        } ?: error("Couldn’t read that identity photo.")
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Couldn’t decode that identity photo." }
        val needsSizing = metadata.sizeBytes !in 0..(2L * 1024 * 1024) ||
            maxOf(bounds.outWidth, bounds.outHeight) > 4096
        val bytes = if (needsSizing) normalizePhoto(uri, metadata.mimeType, bounds) else {
            context.contentResolver.openInputStream(uri)?.use { input ->
                readIdentityPhotoBytes(input, metadata.sizeBytes)
            } ?: error("Couldn’t read that identity photo.")
        }
        val mimeType = if (needsSizing) "image/png" else metadata.mimeType
        val originalName = File(metadata.filename).name.ifBlank {
            if (metadata.mimeType == "image/jpeg") "identity.jpg" else "identity.png"
        }
        return AndroidIdentityPhotoResult(
            filename = if (needsSizing) "${originalName.substringBeforeLast('.', originalName)}.png" else originalName,
            mimeType = mimeType,
            sizeBytes = bytes.size.toLong(),
            dataB64 = Base64.encodeToString(bytes, Base64.NO_WRAP),
        )
    }

    private fun normalizePhoto(uri: Uri, mimeType: String, bounds: BitmapFactory.Options): ByteArray {
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight).toLong() > 4096L * sample) sample *= 2
        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        var bitmap = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        } ?: error("Couldn’t decode that identity photo.")
        try {
            // The sampled bitmap has no EXIF metadata: apply JPEG orientation before encoding.
            val orientation = if (mimeType == "image/jpeg") {
                context.contentResolver.openInputStream(uri)?.use {
                    ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
                } ?: ExifInterface.ORIENTATION_NORMAL
            } else ExifInterface.ORIENTATION_NORMAL
            val matrix = Matrix().apply {
                when (orientation) {
                    ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> setScale(-1f, 1f)
                    ExifInterface.ORIENTATION_ROTATE_180 -> setRotate(180f)
                    ExifInterface.ORIENTATION_FLIP_VERTICAL -> setScale(1f, -1f)
                    ExifInterface.ORIENTATION_TRANSPOSE -> { setRotate(90f); postScale(-1f, 1f) }
                    ExifInterface.ORIENTATION_ROTATE_90 -> setRotate(90f)
                    ExifInterface.ORIENTATION_TRANSVERSE -> { setRotate(-90f); postScale(-1f, 1f) }
                    ExifInterface.ORIENTATION_ROTATE_270 -> setRotate(-90f)
                }
            }
            if (!matrix.isIdentity) {
                val oriented = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                if (oriented !== bitmap) { bitmap.recycle(); bitmap = oriented }
            }
            while (true) {
                val output = ByteArrayOutputStream()
                check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)) { "Couldn’t encode that identity photo." }
                val bytes = output.toByteArray()
                if (bytes.size <= 2 * 1024 * 1024) return bytes
                val width = maxOf(1, bitmap.width * 3 / 4)
                val height = maxOf(1, bitmap.height * 3 / 4)
                check(width < bitmap.width || height < bitmap.height) { "Couldn’t fit that identity photo." }
                val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
                bitmap.recycle()
                bitmap = scaled
            }
        } finally {
            bitmap.recycle()
        }
    }

    private fun queryMetadata(uri: Uri): IdentityMetadata {
        var filename: String? = null
        var size = -1L
        context.contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE),
            null,
            null,
            null,
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (nameIndex >= 0 && !cursor.isNull(nameIndex)) filename = cursor.getString(nameIndex)
                if (sizeIndex >= 0 && !cursor.isNull(sizeIndex)) size = cursor.getLong(sizeIndex)
            }
        }
        if (size < 0) {
            size = context.contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1
        }
        val mimeType = context.contentResolver.getType(uri)?.lowercase()?.substringBefore(';')
            ?: when (filename?.substringAfterLast('.', "")?.lowercase()) {
                "jpg", "jpeg" -> "image/jpeg"
                "png" -> "image/png"
                else -> ""
            }
        return IdentityMetadata(filename.orEmpty(), mimeType, size)
    }

    data class CameraTarget(val file: File, val uri: Uri)
    private data class IdentityMetadata(val filename: String, val mimeType: String, val sizeBytes: Long)
}
