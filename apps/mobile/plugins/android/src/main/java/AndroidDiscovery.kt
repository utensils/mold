package com.utensils.mold.mobile_native

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.net.wifi.WifiManager
import android.os.Handler
import android.os.Looper
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import org.json.JSONArray
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicBoolean

internal class AndroidDiscovery(
    context: Context,
    private val timeoutMs: Long,
    private val invoke: Invoke,
) {
    private val appContext = context.applicationContext
    private val manager = appContext.getSystemService(Context.NSD_SERVICE) as NsdManager
    private val handler = Handler(Looper.getMainLooper())
    private val services = linkedMapOf<String, NsdServiceInfo>()
    private val pending = ArrayDeque<NsdServiceInfo>()
    private val hosts = linkedMapOf<String, DiscoveredService>()
    private val finished = AtomicBoolean(false)
    private var discoveryStarted = false
    private var resolving = false
    private var multicastLock: WifiManager.MulticastLock? = null

    fun start() {
        handler.post {
            try {
                acquireMulticastLock()
                handler.postDelayed(::stopAndResolve, timeoutMs)
                manager.discoverServices(SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, listener)
            } catch (error: Exception) {
                reject("could not start nearby discovery: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    private val listener = object : NsdManager.DiscoveryListener {
        override fun onDiscoveryStarted(serviceType: String) {
            discoveryStarted = true
        }

        override fun onServiceFound(serviceInfo: NsdServiceInfo) {
            if (serviceInfo.serviceType.startsWith("_mold._tcp")) {
                services["${serviceInfo.serviceName}|${serviceInfo.serviceType}"] = serviceInfo
            }
        }

        override fun onServiceLost(serviceInfo: NsdServiceInfo) {
            services.remove("${serviceInfo.serviceName}|${serviceInfo.serviceType}")
        }

        override fun onDiscoveryStopped(serviceType: String) {
            discoveryStarted = false
            beginResolving()
        }

        override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) {
            reject("nearby discovery could not start (Android NSD error $errorCode)")
        }

        override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) {
            discoveryStarted = false
            beginResolving()
        }
    }

    private fun stopAndResolve() {
        if (finished.get()) return
        handler.removeCallbacksAndMessages(null)
        if (discoveryStarted) {
            try {
                manager.stopServiceDiscovery(listener)
                return
            } catch (_: Exception) {
                discoveryStarted = false
            }
        }
        beginResolving()
    }

    private fun beginResolving() {
        if (finished.get() || resolving) return
        resolving = true
        pending.addAll(services.values)
        resolveNext()
    }

    @Suppress("DEPRECATION")
    private fun resolveNext() {
        if (finished.get()) return
        val service = pending.pollFirst()
        if (service == null) {
            resolve()
            return
        }
        try {
            manager.resolveService(service, object : NsdManager.ResolveListener {
                override fun onServiceResolved(resolved: NsdServiceInfo) {
                    val address = resolved.host?.hostAddress.orEmpty()
                    val port = resolved.port
                    if (address.isNotBlank() && port in 1..65535) {
                        val connectableAddress = if (address.contains(':')) "[$address]" else address
                        hosts["$connectableAddress:$port"] = DiscoveredService(
                            resolved.serviceName.ifBlank { connectableAddress },
                            connectableAddress,
                            port,
                        )
                    }
                    resolveNext()
                }

                override fun onResolveFailed(serviceInfo: NsdServiceInfo, errorCode: Int) {
                    resolveNext()
                }
            })
        } catch (_: Exception) {
            resolveNext()
        }
    }

    private fun resolve() {
        if (!finished.compareAndSet(false, true)) return
        cleanup()
        val array = JSONArray()
        hosts.values.sortedBy { it.name.lowercase() }.forEach { host ->
            array.put(JSObject().apply {
                put("name", host.name)
                put("host", host.host)
                put("port", host.port)
            })
        }
        invoke.resolve(JSObject().apply { put("hosts", array) })
    }

    private fun reject(message: String) {
        if (!finished.compareAndSet(false, true)) return
        cleanup()
        invoke.reject(message)
    }

    private fun acquireMulticastLock() {
        val wifi = appContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager ?: return
        multicastLock = wifi.createMulticastLock("mold-nearby-discovery").apply {
            setReferenceCounted(false)
            acquire()
        }
    }

    private fun cleanup() {
        handler.removeCallbacksAndMessages(null)
        if (discoveryStarted) {
            try {
                manager.stopServiceDiscovery(listener)
            } catch (_: Exception) {
                // Discovery is already stopping or stopped.
            }
            discoveryStarted = false
        }
        multicastLock?.let { lock -> if (lock.isHeld) lock.release() }
        multicastLock = null
    }

    private data class DiscoveredService(val name: String, val host: String, val port: Int)

    companion object {
        private const val SERVICE_TYPE = "_mold._tcp."
    }
}
