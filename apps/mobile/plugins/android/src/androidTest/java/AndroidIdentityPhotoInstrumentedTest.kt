package com.utensils.mold.mobile_native

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.ExifInterface
import android.graphics.Color
import androidx.core.content.FileProvider
import java.io.File
import android.os.Build
import android.provider.MediaStore
import android.util.Base64
import androidx.test.core.app.ApplicationProvider
import java.io.ByteArrayInputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidIdentityPhotoInstrumentedTest {
    private val context: Context = ApplicationProvider.getApplicationContext()
    private val identity = AndroidIdentityPhoto(context)

    @Test
    fun usesPhotoPickerWithoutBroadStorageAccessOnModernAndroid() {
        val intent = identity.libraryIntent(Build.VERSION_CODES.TIRAMISU)

        assertEquals(MediaStore.ACTION_PICK_IMAGES, intent.action)
        assertEquals("image/*", intent.type)
    }

    @Test
    fun fallsBackToTheDocumentPickerBeforeAndroidPhotoPicker() {
        val intent = identity.libraryIntent(Build.VERSION_CODES.S_V2)

        assertEquals(Intent.ACTION_OPEN_DOCUMENT, intent.action)
        assertTrue(intent.categories.contains(Intent.CATEGORY_OPENABLE))
    }

    @Test
    fun boundsTheStreamEvenWhenAProviderReportsASmallerSize() {
        val bytes = ByteArray((IDENTITY_PHOTO_MAX_BYTES + 1).toInt())

        val error = runCatching {
            readIdentityPhotoBytes(ByteArrayInputStream(bytes), 1)
        }.exceptionOrNull()

        assertTrue(error?.message?.contains("16 MiB") == true)
    }

    @Test
    fun largeCameraPhotoIsDownscaledBeforeBridgeSizeLimits() {
        val target = identity.createCameraTarget()
        val bitmap = Bitmap.createBitmap(5000, 10, Bitmap.Config.ARGB_8888)
        target.file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        bitmap.recycle()
        ExifInterface(target.file.path).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90.toString())
            saveAttributes()
        }
        val result = identity.readPicked(target.uri, target.file)
        val bytes = Base64.decode(result.dataB64, Base64.NO_WRAP)
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        assertTrue(maxOf(decoded.width, decoded.height) <= 4096)
        assertTrue(decoded.height > decoded.width)
        assertTrue(bytes.size <= 2 * 1024 * 1024)
        assertEquals("image/png", result.mimeType)
        assertTrue(result.filename.endsWith(".png"))
        assertEquals(bytes.size.toLong(), result.sizeBytes)
        decoded.recycle()
        target.file.delete()
    }

    @Test
    fun downscaledPngKeepsTransparency() {
        val file = File(context.cacheDir, "identity/transparent.png")
        file.parentFile!!.mkdirs()
        val bitmap = Bitmap.createBitmap(5000, 4, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.TRANSPARENT)
        file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bitmap.recycle()
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.mold-mobile-native-fileprovider", file)
        val result = identity.readPicked(uri)
        val bytes = Base64.decode(result.dataB64, Base64.NO_WRAP)
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        assertTrue(decoded.width <= 4096)
        assertEquals(0, Color.alpha(decoded.getPixel(0, 0)))
        decoded.recycle()
        file.delete()
    }

    @Test
    fun cameraPhotoRetainsExactBytesAndUsesAContentUri() {
        val target = identity.createCameraTarget()
        target.file.outputStream().use { output ->
            Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
                .compress(Bitmap.CompressFormat.JPEG, 95, output)
        }
        val bytes = target.file.readBytes()

        val result = identity.readPicked(target.uri, target.file)

        assertEquals("content", target.uri.scheme)
        assertEquals(target.file.name, result.filename)
        assertEquals(bytes.size.toLong(), result.sizeBytes)
        assertEquals(Base64.encodeToString(bytes, Base64.NO_WRAP), result.dataB64)
        target.file.delete()
    }
}
