package com.utensils.mold.mobile_native

import android.app.Activity
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
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

@TauriPlugin
class MoldMobileNativePlugin(activity: Activity) : Plugin(activity) {
    private val vault = CredentialVault(activity.applicationContext)

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

    private inline fun resolveOrReject(invoke: Invoke, action: String, block: () -> Unit) {
        try {
            block()
        } catch (error: Exception) {
            invoke.reject("could not $action: ${error.message ?: error.javaClass.simpleName}")
        }
    }
}
