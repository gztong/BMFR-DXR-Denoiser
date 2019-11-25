## Real-Time Path Tracing Reconstruction In DXR

### Team member:

Gangzheng Tong

Jiangping Xu

Tianming Xu

### Overview:

This project is complete DXR pipeline that use 1 spp path tracing noisy image and Blockwise Multi-Order Feature Regression technique to produce high resolution physically-correct image. There are five stages in our pipeline:

1. A basic path tracer produces 1 spp output image and populates necessary feature buffers(normal and world position) based on the 1 spp image
2. Pre-process the current frame to mix up with previous frames’ data
3. Apply Blockwise Multi-Order Feature Regression on the feature buffers
4. Post-process the frames’ data by accumulating the current frame’s information in
5. Apply the computed color to the pixel



### Build:

Clone

Falcor -> run [update_dependencies.bat](https://github.com/gztong/BMFR-DXR-Denoiser/blob/master/Falcor/update_dependencies.bat) -> Copy **dear_imgui_addons** into **Framework/Externals**





----------------------------------------------------------------------------------------------
Acknowledgments:
----------------------------------------------------------------------------------------------

​	The base code of path tracer is based on a very good DXR tutorial:  http://cwyman.org/code/dxrTutors/dxr_tutors.md.html, which teaches people how to develop path tracer using a very powerful library Falcor( https://developer.nvidia.com/falcor )