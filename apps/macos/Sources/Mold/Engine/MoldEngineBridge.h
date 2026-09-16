// The C ABI exported by crates/mold-macos-ffi.
//
// Deliberately tiny: nothing about a render crosses this boundary. The app
// reaches the engine over HTTP on loopback, exactly as it reaches a machine on
// the network, so marshalling request types through C would be rebuilding HTTP
// without the HTTP.
#ifndef MOLD_ENGINE_BRIDGE_H
#define MOLD_ENGINE_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

int32_t mold_engine_bootstrap(const char *mold_home, const char *api_key, const char *log_dir);
uint16_t mold_engine_alloc_port(void);
int32_t mold_engine_start(const char *bind, uint16_t port, const char *models_dir);
bool mold_engine_is_alive(void);
bool mold_engine_join(uint64_t timeout_ms);

#endif
