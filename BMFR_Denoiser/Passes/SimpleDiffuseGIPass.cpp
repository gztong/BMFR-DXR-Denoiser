/**********************************************************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#  * Redistributions of code must retain the copyright notice, this list of conditions and the following disclaimer.
#  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT
# SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************************************/

#include "SimpleDiffuseGIPass.h"


// Some global vars, used to simplify changing shader location & entry points
namespace {
	// Where is our shaders located?  -- we will directly find in the Data folder
	const char* kFileRayTrace = "simpleDiffuseGI.rt.hlsl";

	// What are the entry points in that shader for various ray tracing shaders?
	const char* kEntryPointRayGen        = "SimpleDiffuseGIRayGen";

	const char* kEntryPointMiss0         = "ShadowMiss";
	const char* kEntryShadowAnyHit       = "ShadowAnyHit";
	const char* kEntryShadowClosestHit   = "ShadowClosestHit";

	const char* kEntryPointMiss1         = "IndirectMiss";
	const char* kEntryIndirectAnyHit     = "IndirectAnyHit";
	const char* kEntryIndirectClosestHit = "IndirectClosestHit";
};

bool SimpleDiffuseGIPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
	// Stash a copy of our resource manager so we can get rendering resources
	mpResManager = pResManager;
	mpResManager->requestTextureResources({ "WorldPosition", "WorldNormal", "MaterialDiffuse" });
	mpResManager->requestTextureResource(ResourceManager::kOutputChannel);
	mpResManager->requestTextureResource(ResourceManager::kEnvironmentMap);

	// Set the default scene to load
	mpResManager->setDefaultSceneName("Data/pink_room/pink_room.fscene");

	// Create our wrapper around a ray tracing pass.  Tell it where our ray generation shader and ray-specific shaders are
	mpRays = RayLaunch::create(kFileRayTrace, kEntryPointRayGen);

	// Add ray type #0 (shadow rays)
	mpRays->addMissShader(kFileRayTrace, kEntryPointMiss0);
	mpRays->addHitShader(kFileRayTrace, kEntryShadowClosestHit, kEntryShadowAnyHit);

	// Add ray type #1 (indirect GI rays)
	mpRays->addMissShader(kFileRayTrace, kEntryPointMiss1);
	mpRays->addHitShader(kFileRayTrace, kEntryIndirectClosestHit, kEntryIndirectAnyHit);

	// Now that we've passed all our shaders in, compile and (if available) setup the scene
	mpRays->compileRayProgram();
	if (mpScene) mpRays->setScene(mpScene);

    //create a drop down list to render diff buffer
    mDisplayableBuffers.push_back({ -1, "< None >" });
    return true;
}

void SimpleDiffuseGIPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
	// Stash a copy of the scene and pass it to our ray tracer (if initialized)
    mpScene = std::dynamic_pointer_cast<RtScene>(pScene);
	if (mpRays) mpRays->setScene(mpScene);
}

void SimpleDiffuseGIPass::renderGui(Gui* pGui)
{
	// Add a toggle to turn on/off shooting of indirect GI rays
	int dirty = 0;
	dirty |= (int)pGui->addCheckBox(mDoDirectShadows ? "Shooting direct shadow rays" : "No direct shadow rays", mDoDirectShadows);
	dirty |= (int)pGui->addCheckBox(mDoIndirectGI ? "Shooting global illumination rays" : "Skipping global illumination", 
		                            mDoIndirectGI);
	dirty |= (int)pGui->addCheckBox(mDoCosSampling ? "Use cosine sampling" : "Use uniform sampling", mDoCosSampling);
	if (dirty) setRefreshFlag();

    pGui->addDropdown("Displayed", mDisplayableBuffers, mSelectedBuffer);
}


void SimpleDiffuseGIPass::execute(RenderContext* pRenderContext)
{
	// Get the output buffer we're writing into
	Texture::SharedPtr pDstTex = mpResManager->getClearedTexture(ResourceManager::kOutputChannel, vec4(0.0f, 0.0f, 0.0f, 0.0f));

	// Do we have all the resources we need to render?  If not, return
	if (!pDstTex || !mpRays || !mpRays->readyToRender()) return;

    //depend on the buffer we choose, we do the ray trace or directly copy the buffer to output Tex
    if (mSelectedBuffer == PATH_TRACE_INDEX)
    {
        //Do Path trace
        // Set our shader variables for the ray generation shader
        auto rayGenVars = mpRays->getRayGenVars();
        rayGenVars["RayGenCB"]["gMinT"] = mpResManager->getMinTDist();
        rayGenVars["RayGenCB"]["gFrameCount"] = mFrameCount++;
        rayGenVars["RayGenCB"]["gDoIndirectGI"] = mDoIndirectGI;
        rayGenVars["RayGenCB"]["gCosSampling"] = mDoCosSampling;
        rayGenVars["RayGenCB"]["gDirectShadow"] = mDoDirectShadows;

        // Pass our G-buffer textures down to the HLSL so we can shade
        rayGenVars["gPos"] = mpResManager->getTexture("WorldPosition");
        rayGenVars["gNorm"] = mpResManager->getTexture("WorldNormal");
        rayGenVars["gDiffuseMatl"] = mpResManager->getTexture("MaterialDiffuse");
        rayGenVars["gOutput"] = pDstTex;

        // Set our environment map texture for indirect rays that miss geometry 
        auto missVars = mpRays->getMissVars(1);       // Remember, indirect rays are ray type #1
        missVars["gEnvMap"] = mpResManager->getTexture(ResourceManager::kEnvironmentMap);

        // Execute our shading pass and shoot indirect rays
        mpRays->execute(pRenderContext, mpResManager->getScreenSize());
    }
    else
    {   //Copy the existing buffer to output
        // Grab our input buffer, as selected by the user from the GUI
        Texture::SharedPtr inTex = mpResManager->getTexture(mSelectedBuffer);

        // If we have selected an invalid texture, clear our output to black and return.
        if (!inTex || mSelectedBuffer == uint32_t(-1))
        {
            pRenderContext->clearRtv(pDstTex->getRTV().get(), vec4(0.0f, 0.0f, 0.0f, 1.0f));
            return;
        }

        // Copy the selected input buffer to our output buffer.
        pRenderContext->blit(inTex->getSRV(), pDstTex->getRTV());
    }
}

void SimpleDiffuseGIPass::pipelineUpdated(ResourceManager::SharedPtr pResManager)
{
    // This method gets called when the pipeline changes.  We ask the resource manager what textures
    //     are available.  We then create a list of these textures to provide to the user via the
    //     GUI window, so they can choose which to display on screen.

    // This only works if we have a valid resource manager
    if (!pResManager) return;
    mpResManager = pResManager;

    // Clear the GUI's list of displayable textures
    mDisplayableBuffers.clear();

    // We're not allowing the user to display the output buffer, so identify that resource
    int32_t outputChannel = mpResManager->getTextureIndex(ResourceManager::kOutputChannel);
    int32_t environmentMap = mpResManager->getTextureIndex(ResourceManager::kEnvironmentMap);

    // Loop over all resources available in the resource manager
    for (uint32_t i = 0; i < mpResManager->getTextureCount(); i++)
    {
        // If this one is the output resource, skip it
        if (i == outputChannel || i == environmentMap) continue;

        // Add the name of this resource to our GUI's list of displayable resources
        mDisplayableBuffers.push_back({ int32_t(i), mpResManager->getTextureName(i) });

        // If our UI currently had an invalid buffer selected, select this valid one now.
        if (mSelectedBuffer == uint32_t(-1)) mSelectedBuffer = i;
    }

    // If there are no valid textures to select, add a "<None>" entry to our list and select it.
    if (mDisplayableBuffers.size() <= 0)
    {
        mDisplayableBuffers.push_back({ -1, "< None >" });
        mSelectedBuffer = uint32_t(-1);
    }

    //add a default layer of choosing the ray trace entry
    mDisplayableBuffers.push_back({ int32_t(PATH_TRACE_INDEX), "Path Trace Result" });
    mSelectedBuffer = uint32_t(PATH_TRACE_INDEX);
}


