//Denoise stage compute shader
Texture2D<float4> gPos; //world position
Texture2D<float4> gNorm;//world normal
Texture2D<float4> gDiffuseMatl; //diffuse material color
Texture2D<float4> gCurFrame; //current output image

float4 main(float2 texC : TEXCOORD, float4 pos : SV_Position) : SV_TARGET0
{
    uint2 pixelPos = (uint2)pos.xy;
    float4 curColor = gCurFrame[pixelPos];

    return curColor;
}