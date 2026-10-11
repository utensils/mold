#ifndef MOLD_SECURE_H
#define MOLD_SECURE_H
#include <stddef.h>
#include <stdint.h>
/* Handles are process-local, thread-safe and never raw pointers. 0 means failure.
 * All pointer/length pairs must name valid memory for the duration of a call.
 * Output caller owns; all byte APIs return length or -1 input, -2 state/handle,
 * -3 authentication, -4 limit/buffer capacity, -5 random-source failure.
 * Session output buffers should be 65535 bytes. Insufficient capacity does not
 * advance state. Once authentication fails, release and establish a NEW session;
 * never retry a mutation on it. Destroy handles on cancellation/shutdown.
 */
uint64_t mold_secure_identity_new(void);
uint64_t mold_secure_identity_restore(const uint8_t *secret, size_t length);
int32_t mold_secure_identity_export(uint64_t handle, uint8_t *output, size_t capacity);
int32_t mold_secure_identity_binding(uint64_t handle, uint8_t *output, size_t capacity);
int32_t mold_secure_identity_sign(uint64_t handle, const uint8_t *input, size_t length, uint8_t *output, size_t capacity);
/* JSON config: initiator, namespace(hex32), host_key/device_key(hex64),
 * optional host_noise_key/device_noise_key(hex64), sid, rid. */
uint64_t mold_secure_session_new(uint64_t identity, const uint8_t *config, size_t length);
int32_t mold_secure_session_handshake_write(uint64_t handle, uint8_t *output, size_t capacity);
int32_t mold_secure_session_handshake_read(uint64_t handle, const uint8_t *input, size_t length);
int32_t mold_secure_session_handshake_hash(uint64_t handle, uint8_t *output, size_t capacity);
int32_t mold_secure_session_ready(uint64_t handle);
int32_t mold_secure_session_seal(uint64_t handle, const uint8_t *input, size_t length, uint8_t *output, size_t capacity);
int32_t mold_secure_session_open(uint64_t handle, const uint8_t *input, size_t length, uint8_t *output, size_t capacity);
void mold_secure_release(uint64_t handle);
/* Pure validation: returns 1 valid, 0 invalid; never alters a session. */
int32_t mold_secure_verify_signature(const uint8_t *public_key, const uint8_t *input, size_t length, const uint8_t *signature);
/* Host-side owner credential: home UTF-8 path, optional legacy candidate.
 * Persists one private per-home winner; never replaces an existing key.
 * Output capacity must be >=4096. Lengths exclude NUL. Returns length or error.
 * No identity/device grant is created or changed. */
int32_t mold_secure_operator_key(const uint8_t *home, size_t home_length, const uint8_t *candidate, size_t candidate_length, uint8_t *output, size_t capacity);
#endif
