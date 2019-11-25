
#include "DenoisePass.h"

namespace {
	// Where is our shaders located?  -- we will directly find in the Data folder
	const char* kDenoiseFragShader = "bmfrDenoise.ps.hlsl";
	const char* kAccumNoisyDataShader = "preprocess.ps.hlsl";
};

BlockwiseMultiOrderFeatureRegression::BlockwiseMultiOrderFeatureRegression(const std::string& bufferToDenoise)
	: ::RenderPass("BMFR Denoise Pass", "BMFR Denoise Options")
{
	mDenoiseChannel = bufferToDenoise;
}

bool BlockwiseMultiOrderFeatureRegression::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
	if (!pResManager) return false;

	// Stash our resource manager; ask for the texture the developer asked us to accumulate
	mpResManager = pResManager;
	mpResManager->requestTextureResource(mDenoiseChannel); //current frame image
	mpResManager->requestTextureResources({ "WorldPosition", "WorldNormal", "MaterialDiffuse" }); //three feature buffers

	mpResManager->requestTextureResource("BMFR_PrevNorm");
	mpResManager->requestTextureResource("BMFR_PrevPos");
	mpResManager->requestTextureResource("BMFR_PrevNoisy");
	mpResManager->requestTextureResource("BMFR_PrevSpp", ResourceFormat::R32Uint);

	mpResManager->requestTextureResource("BMFR_CurNorm");
	mpResManager->requestTextureResource("BMFR_CurPos");
	//mpResManager->requestTextureResource(mDenoiseChannel); //current frame image

	mpResManager->requestTextureResource("BMFR_CurSpp", ResourceFormat::R32Uint);

	// Create our graphics state and accumulation shader
	mpGfxState = GraphicsState::create();


	mpDenoiseShader = FullscreenLaunch::create(kDenoiseFragShader);
	mpReprojection = FullscreenLaunch::create(kAccumNoisyDataShader);

	// Our GUI needs less space than other passes, so shrink the GUI window.
	setGuiSize(ivec2(250, 135));

	return true;
}

void BlockwiseMultiOrderFeatureRegression::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
	// When our renderer moves around, we want to reset accumulation
	mpScene = pScene;
	mAccumCount = 0;


}

void BlockwiseMultiOrderFeatureRegression::resize(uint32_t width, uint32_t height)
{
	// We need a framebuffer to attach to our graphics pipe state (when running our full-screen pass)
	mpInternalFbo = ResourceManager::createFbo(width, height, ResourceFormat::RGBA32Float);
	mpGfxState->setFbo(mpInternalFbo);

	mpCurReprojFbo = ResourceManager::createFbo(width, height, ResourceFormat::RGBA32Float);
	mpPrevReprojFbo = ResourceManager::createFbo(width, height, ResourceFormat::RGBA32Float);

	mNeedFboClear = true;
	mAccumCount = 0;

}


void BlockwiseMultiOrderFeatureRegression::clearFbos(RenderContext* pCtx)
{
	// Clear our FBOs
	pCtx->clearFbo(mpPrevReprojFbo.get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::All);

	mNeedFboClear = false;

}

void BlockwiseMultiOrderFeatureRegression::renderGui(Gui* pGui)
{
	int dirty = 0;
	dirty |= (int)pGui->addCheckBox(mDoDenoise ? "Do BMFR Denoise" : "Ignore the denoise stage", mDoDenoise);

	if (dirty) setRefreshFlag();
}


void BlockwiseMultiOrderFeatureRegression::execute(RenderContext* pRenderContext)
{
	// Ensure we have received information about our rendering state, or we can't render.
	if (!mpResManager) return;

	// Grab the texture to accumulate
	Texture::SharedPtr inputTexture = mpResManager->getTexture(mDenoiseChannel);

	// If our input texture is invalid, or we've been asked to skip accumulation, do nothing.
	if (!inputTexture || !mDoDenoise) return;

	if (mNeedFboClear) clearFbos(pRenderContext);
	mpReprojection->setCamera(mpScene->getActiveCamera());


	mInputTex.curPos = mpResManager->getTexture("WorldPosition");
	mInputTex.curNorm = mpResManager->getTexture("WorldNormal");
	mInputTex.curNoisy = mpResManager->getTexture(mDenoiseChannel);
	mInputTex.curSpp = mpResManager->getTexture("BMFR_CurSpp");

	mInputTex.prevPos = mpResManager->getTexture("BMFR_PrevPos");
	mInputTex.prevNorm = mpResManager->getTexture("BMFR_PrevNorm");
	mInputTex.prevNoisy = mpResManager->getTexture("BMFR_PrevNoisy");
	mInputTex.prevSpp = mpResManager->getTexture("BMFR_PrevSpp");

	auto denoiseShaderVars = mpDenoiseShader->getVars();
	denoiseShaderVars["PerFrameCB"]["gAccumCount"] = mAccumCount++;

	//pass four variables in, world pos, world normal, diffuse color and curr frame image
	denoiseShaderVars["gPos"] = mpResManager->getTexture("WorldPosition");
	denoiseShaderVars["gNorm"] = mpResManager->getTexture("WorldNormal");
	denoiseShaderVars["gDiffuseMatl"] = mpResManager->getTexture("MaterialDiffuse");

	// Peform BMFR
	accumulate_noisy_data(pRenderContext);

	//// Do the accumulatione
	//mpDenoiseShader->execute(pRenderContext, mpGfxState);
	//// We've accumulated our result.  Copy that back to the input/output buffer
	//pRenderContext->blit(mpInternalFbo->getColorTexture(0)->getSRV(), inputTexture->getRTV());

	pRenderContext->blit(mpCurReprojFbo->getColorTexture(0)->getSRV(), inputTexture->getRTV());

	// Swap resources so we're ready for next frame.
	//std::swap(mpCurReprojFbo, mpPrevReprojFbo);

	pRenderContext->blit(mInputTex.curNoisy->getSRV(), mInputTex.prevNoisy->getRTV());
	pRenderContext->blit(mInputTex.curNorm->getSRV(), mInputTex.prevNorm->getRTV());
	pRenderContext->blit(mInputTex.curPos->getSRV(), mInputTex.prevPos->getRTV());
	pRenderContext->blit(mInputTex.curSpp->getSRV(), mInputTex.prevSpp->getRTV());

}

void BlockwiseMultiOrderFeatureRegression::accumulate_noisy_data(RenderContext* pRenderContext)
{
	// Setup textures for our accumulate_noisy_data shader pass
	auto mpReprojectionVars = mpReprojection->getVars();
	mpReprojectionVars["gCurPos"] = mInputTex.curPos;
	mpReprojectionVars["gCurNorm"] = mInputTex.curNorm;
	mpReprojectionVars["gCurNoisy"] = mInputTex.curNoisy;
	mpReprojectionVars["gCurSpp"] = mInputTex.curSpp;

	mpReprojectionVars["gPrevPos"] = mInputTex.prevPos;
	mpReprojectionVars["gPrevNorm"] = mInputTex.prevNorm;
	mpReprojectionVars["gPrevNoisy"] = mInputTex.prevNoisy;
	mpReprojectionVars["gPrevSpp"] = mInputTex.prevSpp;


	// Setup variables for our accumulate_noisy_data pass
	mpReprojectionVars["PerFrameCB"]["frame_number"] = mAccumCount++;


	// Execute the accumulate_noisy_data pass
	mpGfxState->setFbo(mpCurReprojFbo);
	mpReprojection->execute(pRenderContext, mpGfxState);

}