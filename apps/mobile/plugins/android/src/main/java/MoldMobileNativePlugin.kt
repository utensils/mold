package com.utensils.mold.mobile_native

import android.Manifest
import android.app.Activity
import android.content.res.Configuration
import android.content.Intent
import android.os.Build
import androidx.activity.result.ActivityResult
import app.tauri.PermissionState
import app.tauri.annotation.Command
import app.tauri.annotation.ActivityCallback
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.Permission
import app.tauri.annotation.PermissionCallback
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import androidx.core.view.WindowCompat
import org.json.JSONObject

@InvokeArg
class HostKeyArgs {
    lateinit var hostId: String
}

@InvokeArg
class SetApiKeyArgs {
    lateinit var hostId: String
    lateinit var apiKey: String
}

@InvokeArg
class DiscoveryArgs {
    var timeoutMs: Long = 2500
}

@InvokeArg
class ImageDataArgs {
    lateinit var dataB64: String
}

@InvokeArg
class VideoUrlArgs {
    lateinit var url: String
}

@InvokeArg
class ShareAnimationArgs {
    lateinit var url: String
    var apiKey: String? = null
    lateinit var requestJson: String
    lateinit var filename: String
    lateinit var reuseKey: String
}

@InvokeArg
class AppearanceArgs {
    lateinit var appearance: String
}

@InvokeArg
class IdentityPhotoArgs {
    lateinit var source: String
}

private const val LEGACY_MEDIA_WRITE = "legacyMediaWrite"

private sealed interface PendingLegacyMedia {
    data class Image(val dataB64: String) : PendingLegacyMedia
    data class Video(val url: String) : PendingLegacyMedia
}

@TauriPlugin(
    permissions = [
        Permission(
            strings = [Manifest.permission.WRITE_EXTERNAL_STORAGE],
            alias = LEGACY_MEDIA_WRITE,
        ),
    ],
)
class MoldMobileNativePlugin(private val hostActivity: Activity) : Plugin(hostActivity) {
    private val vault = CredentialVault(hostActivity.applicationContext)
    private val media = AndroidMedia(hostActivity.applicationContext)
    private val identityPhoto = AndroidIdentityPhoto(hostActivity.applicationContext)
    private var pendingIdentityCamera: AndroidIdentityPhoto.CameraTarget? = null
    private var pendingLegacyMedia: PendingLegacyMedia? = null

    @Command
    fun setApiKey(invoke: Invoke) {
        val args = invoke.parseArgs(SetApiKeyArgs::class.java)
        resolveOrReject(invoke, "save API key") {
            vault.set(args.hostId, args.apiKey)
            invoke.resolve()
        }
    }

    @Command
    fun getApiKey(invoke: Invoke) {
        val args = invoke.parseArgs(HostKeyArgs::class.java)
        resolveOrReject(invoke, "read API key") {
            val response = JSObject()
            response.put("apiKey", vault.get(args.hostId) ?: JSONObject.NULL)
            invoke.resolve(response)
        }
    }

    @Command
    fun deleteApiKey(invoke: Invoke) {
        val args = invoke.parseArgs(HostKeyArgs::class.java)
        resolveOrReject(invoke, "delete API key") {
            vault.delete(args.hostId)
            invoke.resolve()
        }
    }

    @Command
    fun discoverMoldHosts(invoke: Invoke) {
        val args = invoke.parseArgs(DiscoveryArgs::class.java)
        AndroidDiscovery(hostActivity, args.timeoutMs.coerceIn(500, 10_000), invoke).start()
    }

    @Command
    fun copyImageToClipboard(invoke: Invoke) {
        val args = invoke.parseArgs(ImageDataArgs::class.java)
        runAsync(invoke, "copy image") {
            val clip = media.prepareImageClip(args.dataB64)
            hostActivity.runOnUiThread {
                resolveOrReject(invoke, "copy image") {
                    media.copyPreparedImage(clip)
                    invoke.resolve()
                }
            }
        }
    }

    @Command
    fun saveImageToPhotos(invoke: Invoke) {
        val args = invoke.parseArgs(ImageDataArgs::class.java)
        runWithLegacyMediaPermission(invoke, PendingLegacyMedia.Image(args.dataB64))
    }

    @Command
    fun saveVideoToPhotos(invoke: Invoke) {
        val args = invoke.parseArgs(VideoUrlArgs::class.java)
        runWithLegacyMediaPermission(invoke, PendingLegacyMedia.Video(args.url))
    }

    @PermissionCallback
    fun legacyMediaWritePermissionCallback(invoke: Invoke) {
        val pending = synchronized(this) {
            pendingLegacyMedia.also { pendingLegacyMedia = null }
        } ?: run {
            invoke.reject("no media save is waiting for Photos access")
            return
        }
        if (getPermissionState(LEGACY_MEDIA_WRITE) !== PermissionState.GRANTED) {
            invoke.reject("Photos access is required to save media on this Android version")
            return
        }
        saveMedia(invoke, pending)
    }

    @Command
    fun shareExportedAnimation(invoke: Invoke) {
        val args = invoke.parseArgs(ShareAnimationArgs::class.java)
        Thread {
            try {
                val chooser = media.prepareAnimationShare(
                    args.url,
                    args.apiKey,
                    args.requestJson,
                    args.filename,
                    args.reuseKey,
                )
                hostActivity.runOnUiThread {
                    resolveOrReject(invoke, "open Android share sheet") {
                        hostActivity.startActivity(chooser)
                        val response = JSObject().apply { put("outcome", "shared") }
                        invoke.resolve(response)
                    }
                }
            } catch (error: Exception) {
                invoke.reject(
                    "could not share animation: ${error.message ?: error.javaClass.simpleName}",
                )
            }
        }.start()
    }

    @Command
    fun pickIdentityPhoto(invoke: Invoke) {
        val source = invoke.parseArgs(IdentityPhotoArgs::class.java).source
        val intent = when (source) {
            "library" -> {
                pendingIdentityCamera = null
                identityPhoto.libraryIntent()
            }
            "camera" -> {
                val target = identityPhoto.createCameraTarget()
                pendingIdentityCamera = target
                identityPhoto.cameraIntent(target)
            }
            else -> {
                invoke.reject("unknown identity photo source $source")
                return
            }
        }
        if (intent.resolveActivity(hostActivity.packageManager) == null) {
            pendingIdentityCamera?.file?.delete()
            pendingIdentityCamera = null
            invoke.reject("No Android app can open that identity photo source.")
            return
        }
        startActivityForResult(invoke, intent, "identityPhotoResult")
    }

    @ActivityCallback
    fun identityPhotoResult(invoke: Invoke, result: ActivityResult) {
        val camera = pendingIdentityCamera.also { pendingIdentityCamera = null }
        if (result.resultCode == Activity.RESULT_CANCELED) {
            camera?.file?.delete()
            invoke.resolve(JSObject().apply { put("cancelled", true) })
            return
        }
        if (result.resultCode != Activity.RESULT_OK) {
            camera?.file?.delete()
            invoke.reject("Android could not pick that identity photo.")
            return
        }
        val uri = camera?.uri ?: result.data?.data
        if (uri == null) {
            camera?.file?.delete()
            invoke.reject("Android returned no identity photo.")
            return
        }
        runAsync(invoke, "read identity photo") {
            try {
                val photo = identityPhoto.readPicked(uri, camera?.file)
                invoke.resolve(JSObject().apply {
                    put("cancelled", false)
                    put("filename", photo.filename)
                    put("mimeType", photo.mimeType)
                    put("sizeBytes", photo.sizeBytes)
                    put("dataB64", photo.dataB64)
                })
            } finally {
                camera?.file?.delete()
            }
        }
    }

    @Command
    fun setMobileAppearance(invoke: Invoke) {
        val args = invoke.parseArgs(AppearanceArgs::class.java)
        hostActivity.runOnUiThread {
            resolveOrReject(invoke, "update appearance") {
                val dark = when (args.appearance) {
                    "dark" -> true
                    "light" -> false
                    "system" -> (hostActivity.resources.configuration.uiMode and
                        Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES
                    else -> throw IllegalArgumentException("unknown appearance ${args.appearance}")
                }
                WindowCompat.getInsetsController(
                    hostActivity.window,
                    hostActivity.window.decorView,
                ).apply {
                    isAppearanceLightStatusBars = !dark
                    isAppearanceLightNavigationBars = !dark
                }
                invoke.resolve()
            }
        }
    }

    private fun runAsync(invoke: Invoke, action: String, block: () -> Unit) {
        Thread {
            try {
                block()
            } catch (error: Exception) {
                invoke.reject("could not $action: ${error.message ?: error.javaClass.simpleName}")
            }
        }.start()
    }

    private fun runWithLegacyMediaPermission(invoke: Invoke, pending: PendingLegacyMedia) {
        if (!needsLegacyMediaWritePermission(Build.VERSION.SDK_INT)) {
            saveMedia(invoke, pending)
            return
        }
        synchronized(this) {
            if (pendingLegacyMedia != null) {
                invoke.reject("another media save is waiting for Photos access")
                return
            }
            pendingLegacyMedia = pending
        }
        try {
            requestPermissionForAlias(
                LEGACY_MEDIA_WRITE,
                invoke,
                "legacyMediaWritePermissionCallback",
            )
        } catch (error: Exception) {
            synchronized(this) { pendingLegacyMedia = null }
            invoke.reject(
                "could not request Photos access: ${error.message ?: error.javaClass.simpleName}",
            )
        }
    }

    private fun saveMedia(invoke: Invoke, pending: PendingLegacyMedia) {
        when (pending) {
            is PendingLegacyMedia.Image -> runAsync(invoke, "save image") {
                media.saveImage(pending.dataB64)
                invoke.resolve()
            }
            is PendingLegacyMedia.Video -> runAsync(invoke, "save video") {
                media.saveVideo(pending.url)
                invoke.resolve()
            }
        }
    }

    private inline fun resolveOrReject(invoke: Invoke, action: String, block: () -> Unit) {
        try {
            block()
        } catch (error: Exception) {
            invoke.reject("could not $action: ${error.message ?: error.javaClass.simpleName}")
        }
    }
}

internal fun needsLegacyMediaWritePermission(sdkInt: Int): Boolean =
    sdkInt <= Build.VERSION_CODES.P
