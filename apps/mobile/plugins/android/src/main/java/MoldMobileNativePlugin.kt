package com.utensils.mold.mobile_native

import android.app.Activity
import android.content.res.Configuration
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
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

@TauriPlugin
class MoldMobileNativePlugin(private val hostActivity: Activity) : Plugin(hostActivity) {
    private val vault = CredentialVault(hostActivity.applicationContext)
    private val media = AndroidMedia(hostActivity.applicationContext)

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
        hostActivity.runOnUiThread {
            resolveOrReject(invoke, "copy image") {
                media.copyImage(args.dataB64)
                invoke.resolve()
            }
        }
    }

    @Command
    fun saveImageToPhotos(invoke: Invoke) {
        val args = invoke.parseArgs(ImageDataArgs::class.java)
        runAsync(invoke, "save image") {
            media.saveImage(args.dataB64)
            invoke.resolve()
        }
    }

    @Command
    fun saveVideoToPhotos(invoke: Invoke) {
        val args = invoke.parseArgs(VideoUrlArgs::class.java)
        runAsync(invoke, "save video") {
            media.saveVideo(args.url)
            invoke.resolve()
        }
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

    private inline fun resolveOrReject(invoke: Invoke, action: String, block: () -> Unit) {
        try {
            block()
        } catch (error: Exception) {
            invoke.reject("could not $action: ${error.message ?: error.javaClass.simpleName}")
        }
    }
}
