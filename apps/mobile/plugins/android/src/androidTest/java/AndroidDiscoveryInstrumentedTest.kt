package com.utensils.mold.mobile_native

import java.net.InetAddress
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AndroidDiscoveryInstrumentedTest {
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
