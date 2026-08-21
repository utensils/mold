package com.utensils.mold.mobile_native

import androidx.test.core.app.ApplicationProvider
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

class CredentialVaultInstrumentedTest {
    private val hostId = "uat-host"
    private val vault = CredentialVault(ApplicationProvider.getApplicationContext())

    @After
    fun cleanUp() {
        vault.delete(hostId)
    }

    @Test
    fun roundTripsWithoutPersistingPlaintext() {
        val apiKey = "uat-secret-that-must-not-be-plaintext"

        vault.set(hostId, apiKey)

        assertEquals(apiKey, vault.get(hostId))
        assertFalse(vault.storedCiphertext(hostId)!!.contains(apiKey))

        vault.delete(hostId)
        assertNull(vault.get(hostId))
    }

    @Test
    fun discardsMalformedCiphertext() {
        vault.storeCiphertextForTesting(hostId, "not-an-encrypted-payload")

        assertNull(vault.get(hostId))
        assertNull(vault.storedCiphertext(hostId))
    }
}
