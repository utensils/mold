package com.utensils.mold.mobile_native

import android.content.ClipData
import android.content.Context
import android.content.Intent
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

internal fun identityPhotoSizeRefusal(size: Long): String? = when {
    size < 0 -> "Couldn’t verify that identity photo’s size. Choose another photo."
    size > IDENTITY_PHOTO_MAX_BYTES -> "Identity photo must be 16 MiB or smaller."
    else -> null
}

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

    /** Resolves metadata, including the 16 MiB limit, before opening the byte stream. */
    fun readPicked(uri: Uri, cameraFile: File? = null): AndroidIdentityPhotoResult {
        val metadata = if (cameraFile != null) {
            IdentityMetadata(cameraFile.name, "image/jpeg", cameraFile.length())
        } else {
            queryMetadata(uri)
        }
        identityPhotoSizeRefusal(metadata.sizeBytes)?.let { throw IllegalArgumentException(it) }
        require(metadata.mimeType == "image/png" || metadata.mimeType == "image/jpeg") {
            "Identity photo must be a PNG or JPEG image."
        }

        val bytes = context.contentResolver.openInputStream(uri)?.use { input ->
            readIdentityPhotoBytes(input, metadata.sizeBytes)
        } ?: error("Couldn’t read that identity photo.")
        return AndroidIdentityPhotoResult(
            filename = File(metadata.filename).name.ifBlank {
                if (metadata.mimeType == "image/jpeg") "identity.jpg" else "identity.png"
            },
            mimeType = metadata.mimeType,
            sizeBytes = metadata.sizeBytes,
            dataB64 = Base64.encodeToString(bytes, Base64.NO_WRAP),
        )
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
