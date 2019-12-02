#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"           
import ShaderCommon; // Shared shading data structures

#define SECOND_BLEND_ALPHA 0.1f

Texture2D<float4> filtered_frame; // new color from preprocess and filter pass

Texture2D<float4> accumulated_prev_frame; // input
Texture2D<float4> albedo; // input

Texture2D<uint> accept_bools; // is previous sample accepted
Texture2D<float2> in_prev_frame_pixel; // from preprocess

RWTexture2D<float4> accumulated_frame; // output

cbuffer PerFrameCB
{
    uint frame_number;
}


float4 main(float2 texC : TEXCOORD, float4 pos : SV_Position) : SV_TARGET0
{
    const uint2 pixel = (uint2) pos.xy;

	const float4 filterData = filtered_frame[pixel];
	const float3 filtered_color = filterData.xyz;
	const float cusSpp = filterData.w;
	
    float3 prev_color = float3(0.f, 0.f, 0.f);
    float blend_alpha = 1.f;
    const uint accept = accept_bools[pixel];

    if (frame_number > 0)
    {
        const uint accept = accept_bools[pixel];
        if (accept > 0)
        { // If any prev frame sample is accepted
			// Bilinear sampling
            const float2 prev_frame_pixel_f = in_prev_frame_pixel[pixel];
            const int2 prev_frame_pixel = int2(prev_frame_pixel_f);
            const float2 prev_pixel_fract = prev_frame_pixel_f - float2(prev_frame_pixel);
            const float2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;
            float total_weight = 0.f;

			// Accept tells if the sample is acceptable based on world position and normal
            if (accept & 0x01)
            {
                float weight = one_minus_prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
                total_weight += weight;
                prev_color += weight * accumulated_prev_frame[prev_frame_pixel].xyz;
            }

            if (accept & 0x02)
            {
                float weight = prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
                total_weight += weight;
                prev_color += weight * accumulated_prev_frame[prev_frame_pixel + int2(1, 0)].xyz;
            }

            if (accept & 0x04)
            {
                float weight = one_minus_prev_pixel_fract.x * prev_pixel_fract.y;
                total_weight += weight;
                prev_color += weight * accumulated_prev_frame[prev_frame_pixel + int2(0, 1)].xyz;
            }

            if (accept & 0x08)
            {
                float weight = prev_pixel_fract.x * prev_pixel_fract.y;
                total_weight += weight;
                prev_color += weight * accumulated_prev_frame[prev_frame_pixel + int2(1, 1)].xyz;
            }

            if (total_weight > 0.f)
            {
            // Blend_alpha is dymically decided so that the result is average
            // of all samples until the cap defined by SECOND_BLEND_ALPHA is reached
				blend_alpha = 1.f / cusSpp;
                blend_alpha = max(blend_alpha, SECOND_BLEND_ALPHA);

                prev_color /= total_weight;
            }
        }
    }
	// Mix with colors and store results
    float3 accumulated_color = blend_alpha * filtered_color + (1.f - blend_alpha) * prev_color;
    accumulated_frame[pixel] = float4(accumulated_color, 1.f);

    return accumulated_frame[pixel];

}