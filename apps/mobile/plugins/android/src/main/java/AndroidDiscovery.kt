package com.utensils.mold.mobile_native

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import org.json.JSONArray
import java.net.Inet4Address
import java.net.InetAddress
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
            onHandler { discoveryStarted = true }
        }

        override fun onServiceFound(serviceInfo: NsdServiceInfo) {
            onHandler {
                if (serviceInfo.serviceType.startsWith("_mold._tcp")) {
                    services["${serviceInfo.serviceName}|${serviceInfo.serviceType}"] = serviceInfo
                }
            }
        }

        override fun onServiceLost(serviceInfo: NsdServiceInfo) {
            onHandler {
                services.remove("${serviceInfo.serviceName}|${serviceInfo.serviceType}")
            }
        }

        override fun onDiscoveryStopped(serviceType: String) {
            onHandler {
                discoveryStarted = false
                beginResolving()
            }
        }

        override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) {
            onHandler { reject("nearby discovery could not start (Android NSD error $errorCode)") }
        }

        override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) {
            onHandler {
                discoveryStarted = false
                beginResolving()
            }
        }
    }

    private fun stopAndResolve() {
        if (finished.get()) return
        handler.removeCallbacksAndMessages(null)
        if (discoveryStarted) {
            try {
                manager.stopServiceDiscovery(listener)
                handler.postDelayed(::beginResolving, STOP_TIMEOUT_MS)
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
        handler.postDelayed(::resolve, minOf(timeoutMs, RESOLVE_TIMEOUT_MS))
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
                    onHandler {
                        if (finished.get()) return@onHandler
                        val addresses = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                            resolved.hostAddresses
                        } else {
                            listOfNotNull(resolved.host)
                        }
                        val address = selectConnectableAddress(addresses)
                        val port = resolved.port
                        if (address != null && port in 1..65535) {
                            hosts["$address:$port"] = DiscoveredService(
                                resolved.serviceName.ifBlank { address },
                                address,
                                port,
                            )
                        }
                        resolveNext()
                    }
                }

                override fun onResolveFailed(serviceInfo: NsdServiceInfo, errorCode: Int) {
                    onHandler(::resolveNext)
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

    private fun onHandler(block: () -> Unit) {
        if (Looper.myLooper() == handler.looper) block() else handler.post(block)
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
        private const val STOP_TIMEOUT_MS = 500L
        private const val RESOLVE_TIMEOUT_MS = 3_000L
    }
}

internal fun selectConnectableAddress(addresses: List<InetAddress>): String? {
    val selected = addresses.firstOrNull { it is Inet4Address && !it.isAnyLocalAddress }
        ?: addresses.firstOrNull { !it.isAnyLocalAddress && !it.isLinkLocalAddress }
        ?: return null
    val address = selected.hostAddress?.substringBefore('%')?.takeIf { it.isNotBlank() } ?: return null
    return if (address.contains(':')) "[$address]" else address
}
