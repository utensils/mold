package com.utensils.mold.mobile_native

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import androidx.test.core.app.ApplicationProvider
import app.tauri.plugin.Invoke
import com.fasterxml.jackson.databind.ObjectMapper
import org.json.JSONObject
import java.net.ServerSocket
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertTrue
import java.net.InetAddress
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AndroidDiscoveryInstrumentedTest {
    /** Exercise Android NSD and the production discovery bridge with our own
     * ephemeral advertisement. No remote Mold server or generation is involved.
     */
    @Test
    fun discoversRegisteredServiceWithPairingIdentity() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val manager = context.getSystemService(Context.NSD_SERVICE) as NsdManager
        val registered = CountDownLatch(1)
        val removed = CountDownLatch(1)
        val registrationError = AtomicReference<String>()
        val serviceName = "mold-redesign-uat-${System.nanoTime()}"
        val instanceId = "redesign-uat-${System.nanoTime()}"
        val listener = object : NsdManager.RegistrationListener {
            override fun onServiceRegistered(info: NsdServiceInfo) { registered.countDown() }
            override fun onRegistrationFailed(info: NsdServiceInfo, code: Int) {
                registrationError.set("NSD registration failed: $code")
                registered.countDown()
            }
            override fun onServiceUnregistered(info: NsdServiceInfo) { removed.countDown() }
            override fun onUnregistrationFailed(info: NsdServiceInfo, code: Int) {
                registrationError.set("NSD cleanup failed: $code")
                removed.countDown()
            }
        }
        ServerSocket(0).use { server ->
            val service = NsdServiceInfo().apply {
                this.serviceName = serviceName
                serviceType = "_mold._tcp."
                port = server.localPort
                setAttribute("auth", "1")
                setAttribute("id", instanceId)
            }
            manager.registerService(service, NsdManager.PROTOCOL_DNS_SD, listener)
            try {
                assertTrue("NSD registration timed out", registered.await(10, TimeUnit.SECONDS))
                assertNull(registrationError.get())
                val completed = CountDownLatch(1)
                val result = AtomicReference<Pair<Long, String>>()
                val invoke = Invoke(1, "discoverMoldHosts", 2, 3, { callback, data ->
                    result.set(callback to data)
                    completed.countDown()
                }, "{}", ObjectMapper())
                AndroidDiscovery(context, 5_000, invoke).start()
                assertTrue("Discovery did not settle", completed.await(15, TimeUnit.SECONDS))
                assertEquals("Discovery rejected: ${result.get().second}", 2L, result.get().first)
                val hosts = JSONObject(result.get().second).getJSONArray("hosts")
                val found = (0 until hosts.length()).map { hosts.getJSONObject(it) }
                    .firstOrNull { it.optString("instanceId") == instanceId }
                assertTrue("Our registered service was not discovered: $hosts", found != null)
                assertEquals(server.localPort, found!!.getInt("port"))
                assertTrue(found.getBoolean("authRequired"))
                assertTrue(found.getString("name").startsWith(serviceName))
                assertTrue(found.getString("host").isNotBlank())
            } finally {
                manager.unregisterService(listener)
                assertTrue("NSD advertisement cleanup timed out", removed.await(10, TimeUnit.SECONDS))
                assertNull(registrationError.get())
            }
        }
    }

    @Test
    fun prefersIpv4OverIpv6() {
        val addresses = listOf(
            InetAddress.getByName("2001:db8::1"),
            InetAddress.getByName("192.0.2.10"),
        )

        assertEquals("192.0.2.10", selectConnectableAddress(addresses))
    }

    @Test
    fun rejectsLinkLocalIpv6() {
        assertNull(selectConnectableAddress(listOf(InetAddress.getByName("fe80::1"))))
    }

    @Test
    fun bracketsGlobalIpv6() {
        assertEquals("[2001:db8::1]", selectConnectableAddress(listOf(InetAddress.getByName("2001:db8::1"))))
    }
}
