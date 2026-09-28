#include <metal_stdlib>
using namespace metal;

// The mesh shading model, line for line from `studio/components/MeshViewer.vue`
// (GLSL ES 1.00 at :105-158), so the native view and the browser's draw the
// same pixels. Nothing here is a Metal-flavoured re-derivation: the two lights
// are VIEW-space constants, the rim term is the same exponent, the default
// albedo is the same triple, and a back face is lit by its FLIPPED normal
// rather than left black, because extracted surfaces are not reliably closed
// and mold writes `doubleSided` materials.
//
// `packed_float3` and not `float3`: a `float3` in a device buffer has a
// 16-byte stride, and the parser hands over tightly packed xyz triples.

struct MeshUniforms {
    float4x4 modelView;
    float4x4 projection;
    float3x3 normalMatrix;
    float hasTexture;
    float wireframe;
};

struct Varyings {
    float4 position [[position]];
    float3 normal;
    float3 view;
    float3 color;
    float2 uv;
};

vertex Varyings mold_mesh_vertex(uint vertexIndex [[vertex_id]],
                                 const device packed_float3 *positions [[buffer(0)]],
                                 const device packed_float3 *normals [[buffer(1)]],
                                 const device packed_float3 *colors [[buffer(2)]],
                                 const device float2 *uvs [[buffer(3)]],
                                 constant MeshUniforms &uniforms [[buffer(4)]]) {
    Varyings out;
    float4 view = uniforms.modelView * float4(positions[vertexIndex], 1.0);
    out.normal = uniforms.normalMatrix * float3(normals[vertexIndex]);
    out.view = view.xyz;
    out.color = float3(colors[vertexIndex]);
    out.uv = uvs[vertexIndex];
    out.position = uniforms.projection * view;
    return out;
}

fragment float4 mold_mesh_fragment(Varyings in [[stage_in]],
                                   bool isFrontFacing [[front_facing]],
                                   constant MeshUniforms &uniforms [[buffer(0)]],
                                   texture2d<float> baseColour [[texture(0)]]) {
    // The edge pass reuses this function: one flag is cheaper than a second
    // pipeline, and the lines want a flat colour rather than the lighting.
    if (uniforms.wireframe > 0.5) {
        return float4(0.16, 0.85, 0.98, 1.0);
    }
    constexpr sampler baseSampler(filter::linear, mip_filter::none,
                                  address::clamp_to_edge);

    float3 normal = normalize(in.normal);
    if (!isFrontFacing) normal = -normal;
    float3 eye = normalize(-in.view);
    float3 key = normalize(float3(0.45, 0.72, 0.85));
    float3 fill = normalize(float3(-0.65, -0.15, 0.35));
    float kd = max(dot(normal, key), 0.0);
    float fd = max(dot(normal, fill), 0.0);
    float rim = pow(1.0 - max(dot(normal, eye), 0.0), 2.5);
    float3 albedo = in.color;
    if (uniforms.hasTexture > 0.5) albedo *= baseColour.sample(baseSampler, in.uv).rgb;
    float3 lit = albedo * (0.20 + 0.72 * kd + 0.22 * fd) + float3(0.16, 0.19, 0.24) * rim;
    return float4(clamp(lit, 0.0, 1.0), 1.0);
}
