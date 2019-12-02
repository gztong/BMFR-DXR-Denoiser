#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"           

//Denoise stage compute shader
Texture2D<float4> gCurPos; //world position
Texture2D<float4> gPrevPos; //world position

Texture2D<float4> gCurNorm; //world normal
Texture2D<float4> gPrevNorm; //world normal

RWTexture2D<float4> gCurNoisy; //current output image
Texture2D<float4> gPrevNoisy;

RWTexture2D<uint> accept_bools; // is previous sample accepted
RWTexture2D<float2> out_prev_frame_pixel; // save this for later, avoid calculating again

import ShaderCommon; // Shared shading data structures

#define POSITION_LIMIT_SQUARED 0.01f
#define NORMAL_LIMIT_SQUARED 1.0f
#define BLEND_ALPHA 0.2f

cbuffer PerFrameCB
{
    uint frame_number;
}


float4 main(float2 texC : TEXCOORD, float4 pos : SV_Position) : SV_TARGET0
{
    uint2 pixelPos = (uint2) pos.xy;
	if (texC.x > 0.5) return gCurNoisy[pixelPos];

    uint2 dim;
    gCurPos.GetDimensions(dim.x, dim.y);
    uint IMAGE_WIDTH = dim.x;
    uint IMAGE_HEIGHT = dim.y; // TODO: Optimize this

    const int2 pixel = pixelPos;

    float3 world_position = gCurPos[pixelPos].xyz;
    float3 normal = gCurNorm[pixelPos].xyz;
    float3 current_color = gCurNoisy[pixelPos].xyz;

    // Default previous frame pixel is the same pixel
    float2 prev_frame_pixel_f = pos.xy;

	// This is changed to non zero if previous frame is not discarded completely
    uint store_accept = 0x00;

    // Blend_alpha 1.f means that only current frame color is used. The value
    // is changed if sample from previous frame can be used
    float blend_alpha = 1.f;
    float3 previous_color = float3(0.f, 0.f, 0.f);

    float sample_spp = 0.f;
    float total_weight = 0.f;

    if (frame_number > 0)
    {
		// gCamera.prevViewProjMat
		// Matrix multiplication and normalization to 0..1
        float4 prev_frame_pos = mul(float4(world_position, 1.f), gCamera.prevViewProjMat);
        prev_frame_pos /= prev_frame_pos.w;
        prev_frame_pos.x = (prev_frame_pos.x + 1.f) / 2.0f;
        prev_frame_pos.y = (1 - prev_frame_pos.y) / 2.0f;
        float2 prev_frame_uv = prev_frame_pos.xy;

        // Change to pixel indexes and apply offset
        prev_frame_pixel_f = prev_frame_uv * dim;

        float2 pixel_offset = float2(0.500000, 0.500000); // TODO
        prev_frame_pixel_f -= float2(pixel_offset.x, 1 - pixel_offset.y);

        int2 prev_frame_pixel = int2(prev_frame_pixel_f);

      // These are needed for the bilinear sampling
        int2 offsets[4] =
        {
            int2(0, 0),
			int2(1, 0),
			int2(0, 1),
			int2(1, 1),
        };

        float2 prev_pixel_fract = prev_frame_pixel_f - float2(prev_frame_pixel);

        float2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;
        float weights[4];
        weights[0] = one_minus_prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
        weights[1] = prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
        weights[2] = one_minus_prev_pixel_fract.x * prev_pixel_fract.y;
        weights[3] = prev_pixel_fract.x * prev_pixel_fract.y;
        total_weight = 0.f;
      // Bilinear sampling
        for (int i = 0; i < 4; ++i)
        {
            int2 sample_location = prev_frame_pixel + offsets[i];
         // Check if previous frame color can be used based on its screen location
            if (sample_location.x >= 0 && sample_location.y >= 0 &&
					sample_location.x < IMAGE_WIDTH && sample_location.y < IMAGE_HEIGHT)
            {
				// Fetch previous frame world position
                float3 prev_world_position = gPrevPos[sample_location].xyz;

				// Compute world distance squared
                float3 position_difference = prev_world_position - world_position;
                float position_distance_squared = dot(position_difference, position_difference);

                // World position distance discard
                if (position_distance_squared < POSITION_LIMIT_SQUARED)
                {
					// Fetch previous frame normal
                    float3 prev_normal = gPrevNorm[sample_location].xyz;

					// Distance of the normals
                	// NOTE: could use some other distance metric (e.g. angle), but we use hard
					// experimentally found threshold -> means that the metric doesn't matter.
                    float3 normal_difference = prev_normal - normal;
                    float normal_distance_squared = dot(normal_difference, normal_difference);

					// Normal distance discard
                    if (normal_distance_squared < NORMAL_LIMIT_SQUARED)
                    {
						// Pixel passes all tests so store it to accept bools
                        store_accept |= 1 << i;
                        float4 prevData = gPrevNoisy[sample_location];
                        sample_spp += weights[i] * prevData.w;
						
                        previous_color += weights[i] * prevData.xyz;
                        total_weight += weights[i];
                    }
                }
            }
        }
        if (total_weight > 0.f)
        {
            previous_color /= total_weight;
            sample_spp /= total_weight;
           // Blend_alpha is dymically decided so that the result is average
           // of all samples until the cap defined by BLEND_ALPHA is reached
            blend_alpha = 1.f / (sample_spp + 1.f);
            blend_alpha = max(blend_alpha, BLEND_ALPHA);
        }
    } // end if frame_number > 0

   // Store new spp
    float new_spp = 1.f;
    if (blend_alpha < 1.f)
    {
        new_spp += sample_spp;
    }

    float3 new_color = blend_alpha * current_color + (1.f - blend_alpha) * previous_color;

    gCurNoisy[pixelPos] = float4(new_color, new_spp);
    accept_bools[pixelPos] = store_accept;
    out_prev_frame_pixel[pixelPos] = prev_frame_pixel_f;

    return gCurNoisy[pixelPos];
}