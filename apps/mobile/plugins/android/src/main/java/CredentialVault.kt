package com.utensils.mold.mobile_native

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.nio.charset.StandardCharsets
import java.security.KeyStore
import java.security.MessageDigest
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

internal class CredentialVault(private val context: Context) {
    private val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)

    fun set(hostId: String, apiKey: String) {
        require(hostId.isNotBlank()) { "host id is required" }
        require(apiKey.isNotEmpty()) { "API key is required" }

        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        cipher.updateAAD(hostId.toByteArray(StandardCharsets.UTF_8))
        val ciphertext = cipher.doFinal(apiKey.toByteArray(StandardCharsets.UTF_8))
        val payload = listOf(cipher.iv, ciphertext).joinToString(".") {
            Base64.encodeToString(it, Base64.NO_WRAP)
        }
        check(preferences.edit().putString(preferenceKey(hostId), payload).commit()) {
            "encrypted credential could not be committed"
        }
    }

    fun get(hostId: String): String? {
        require(hostId.isNotBlank()) { "host id is required" }
        val payload = preferences.getString(preferenceKey(hostId), null) ?: return null
        val pieces = payload.split('.', limit = 2)
        check(pieces.size == 2) { "encrypted credential is malformed" }

        val cipher = Cipher.getInstance(TRANSFORMATION)
        val iv = Base64.decode(pieces[0], Base64.NO_WRAP)
        val ciphertext = Base64.decode(pieces[1], Base64.NO_WRAP)
        cipher.init(Cipher.DECRYPT_MODE, getOrCreateKey(), GCMParameterSpec(128, iv))
        cipher.updateAAD(hostId.toByteArray(StandardCharsets.UTF_8))
        return String(cipher.doFinal(ciphertext), StandardCharsets.UTF_8)
    }

    fun delete(hostId: String) {
        require(hostId.isNotBlank()) { "host id is required" }
        check(preferences.edit().remove(preferenceKey(hostId)).commit()) {
            "encrypted credential could not be removed"
        }
    }

    internal fun storedCiphertext(hostId: String): String? =
        preferences.getString(preferenceKey(hostId), null)

    private fun getOrCreateKey(): SecretKey {
        val store = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        (store.getKey(KEY_ALIAS, null) as? SecretKey)?.let { return it }

        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        generator.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
                .build(),
        )
        return generator.generateKey()
    }

    private fun preferenceKey(hostId: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(hostId.toByteArray(StandardCharsets.UTF_8))
        return Base64.encodeToString(digest, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
    }

    private companion object {
        const val PREFERENCES = "mold-secure-host-keys-v1"
        const val KEYSTORE = "AndroidKeyStore"
        const val KEY_ALIAS = "com.utensils.mold.remote-api-key.v1"
        const val TRANSFORMATION = "AES/GCM/NoPadding"
    }
}
