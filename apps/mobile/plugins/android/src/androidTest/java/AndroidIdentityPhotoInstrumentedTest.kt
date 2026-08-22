package com.utensils.mold.mobile_native

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Build
import android.provider.MediaStore
import android.util.Base64
import androidx.test.core.app.ApplicationProvider
import java.io.ByteArrayInputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
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
    fun rejectsOversizedMetadataBeforePhotoBytesAreAccepted() {
        assertNull(identityPhotoSizeRefusal(IDENTITY_PHOTO_MAX_BYTES))
        assertTrue(
            identityPhotoSizeRefusal(IDENTITY_PHOTO_MAX_BYTES + 1)!!.contains("16 MiB"),
        )
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
