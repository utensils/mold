// Mold-owned C ABI around pinned xatlas. No inference or Python runtime.
#include "xatlas.h"
#include <cstdint>
#include <memory>

using Continue = bool (*)(const void *);
struct Callback { Continue proceed; const void *state; };
static bool progress(xatlas::ProgressCategory, int, void *data) {
    const auto *callback = static_cast<const Callback *>(data);
    return callback->proceed(callback->state);
}

extern "C" void *mold_xatlas_generate(const float *positions, uint32_t vertex_count,
                                      const uint32_t *indices, uint32_t index_count,
                                      Continue proceed, const void *state,
                                      uint32_t *out_vertices, uint32_t *out_indices) noexcept {
    try {
        Callback callback{proceed, state};
        std::unique_ptr<xatlas::Atlas, decltype(&xatlas::Destroy)> atlas(xatlas::Create(), xatlas::Destroy);
        if (!atlas || !proceed(state)) return nullptr;
        xatlas::SetProgressCallback(atlas.get(), progress, &callback);
        xatlas::MeshDecl mesh;
        mesh.vertexPositionData = positions;
        mesh.vertexPositionStride = sizeof(float) * 3;
        mesh.vertexCount = vertex_count;
        mesh.indexData = indices;
        mesh.indexFormat = xatlas::IndexFormat::UInt32;
        mesh.indexCount = index_count;
        if (xatlas::AddMesh(atlas.get(), mesh) != xatlas::AddMeshError::Success) return nullptr;
        // Match xatlas-python 0.0.9 parametrize: positions and faces only,
        // default ChartOptions/PackOptions, including automatic atlas size.
        xatlas::Generate(atlas.get());
        xatlas::SetProgressCallback(atlas.get());
        if (!proceed(state) || atlas->meshCount != 1 || !atlas->width || !atlas->height) return nullptr;
        const auto &output = atlas->meshes[0];
        if (!output.vertexCount || output.indexCount != index_count) return nullptr;
        for (uint32_t i = 0; i < output.vertexCount; ++i) {
            if (output.vertexArray[i].atlasIndex != 0 || output.vertexArray[i].xref >= vertex_count) return nullptr;
        }
        *out_vertices = output.vertexCount;
        *out_indices = output.indexCount;
        return atlas.release();
    } catch (...) { return nullptr; }
}

extern "C" bool mold_xatlas_copy(const void *handle, uint32_t *mapping, float *uv,
                                  uint32_t *indices, uint32_t vertex_capacity,
                                  uint32_t index_capacity) noexcept {
    const auto *atlas = static_cast<const xatlas::Atlas *>(handle);
    const auto &mesh = atlas->meshes[0];
    if (vertex_capacity != mesh.vertexCount || index_capacity != mesh.indexCount) return false;
    for (uint32_t i = 0; i < mesh.vertexCount; ++i) {
        mapping[i] = mesh.vertexArray[i].xref;
        uv[i * 2] = mesh.vertexArray[i].uv[0] / atlas->width;
        uv[i * 2 + 1] = mesh.vertexArray[i].uv[1] / atlas->height;
    }
    for (uint32_t i = 0; i < mesh.indexCount; ++i) indices[i] = mesh.indexArray[i];
    return true;
}

extern "C" void mold_xatlas_destroy(void *handle) noexcept {
    xatlas::Destroy(static_cast<xatlas::Atlas *>(handle));
}
