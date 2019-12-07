
#include "DenoisePass.h"

namespace {
	// Where is our shaders located?  -- we will directly find in the Data folder
	const char* kDenoiseFragShader = "bmfrDenoise.ps.hlsl";
	const char* kAccumNoisyDataShader = "preprocess.ps.hlsl";
	const char* kRegressionShader = "regressionCP.hlsl";
	const char* kAccumFilteredDataShader = "postprocess.ps.hlsl";
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
	mpResManager->requestTextureResource("BMFR_PrevFiltered");

	mpResManager->requestTextureResource("BMFR_CurNorm");
	mpResManager->requestTextureResource("BMFR_CurPos");
	//mpResManager->requestTextureResource(mDenoiseChannel); //current frame image

	mpResManager->requestTextureResource("BMFR_AcceptedBools", ResourceFormat::R32Uint);
	mpResManager->requestTextureResource("BMFR_PrevFramePixel", ResourceFormat::RG16Float);

	mpResManager->requestTextureResource("BMFR_Output");
	
	int width, height;
	width = (1920 + 31) / 32; // 32 is block edge length
	height = (1055 + 31) / 32;
	int w = width + 1;
	int h = height + 1;
	w /= 2;
	mpResManager->requestTextureResource("tmp_data", ResourceFormat::R32Float, ResourceManager::kDefaultFlags, 1024, w * h * 13);//change texture size if change blocksize and features num
	mpResManager->requestTextureResource("out_data", ResourceFormat::R32Float, ResourceManager::kDefaultFlags, 1024, w * h * 13);//change texture size if change blocksize and features num


	//UnorderedAccessView::SharedPtr uavView = UnorderedAccessView::create(1,0,);

	// Create our graphics state and accumulation shader
	mpGfxState = GraphicsState::create();

	// mpDenoiseShader = FullscreenLaunch::create(kDenoiseFragShader);
	mpPreprocessShader = FullscreenLaunch::create(kAccumNoisyDataShader);
	mpRegression = ComputeProgram::createFromFile("regressionCP.hlsl", "fit");
	mpPostShader = FullscreenLaunch::create(kAccumFilteredDataShader);
	mpCPState = ComputeState::create();
	mpCPState->setProgram(mpRegression);
	mpRegressionVars = ComputeVars::create(mpRegression->getReflector());

	ConstantBuffer::SharedPtr pCB = ConstantBuffer::create((Program::SharedPtr)mpRegression, "PerFrameCB", 128/* 4 * 32 */);
	mpRegressionVars->setConstantBuffer("PerFrameCB", pCB);
	// Our GUI needs less space than other passes, so shrink the GUI window.
	setGuiSize(ivec2(250, 135));


	return true;
}

bool BlockwiseMultiOrderFeatureRegression::initialize(RenderContext * pRenderContext, ResourceManager::SharedPtr pResManager, uint width, uint height)
{
    if (!pResManager) return false;

    // Stash our resource manager; ask for the texture the developer asked us to accumulate
    mpResManager = pResManager;
    mpResManager->requestTextureResource(mDenoiseChannel); //current frame image
    mpResManager->requestTextureResources({ "WorldPosition", "WorldNormal", "MaterialDiffuse" }); //three feature buffers

    mpResManager->requestTextureResource("BMFR_PrevNorm");
    mpResManager->requestTextureResource("BMFR_PrevPos");
    mpResManager->requestTextureResource("BMFR_PrevNoisy");
    mpResManager->requestTextureResource("BMFR_PrevFiltered");

    mpResManager->requestTextureResource("BMFR_CurNorm");
    mpResManager->requestTextureResource("BMFR_CurPos");
    //mpResManager->requestTextureResource(mDenoiseChannel); //current frame image

    mpResManager->requestTextureResource("BMFR_AcceptedBools", ResourceFormat::R32Uint);
    mpResManager->requestTextureResource("BMFR_PrevFramePixel", ResourceFormat::RG16Float);

    mpResManager->requestTextureResource("BMFR_Output");

    int blockWidth, blockHeight;
    blockWidth = (width + 31) / 32; // 32 is block edge length
    blockHeight = (height + 31) / 32;
    int w = blockWidth + 1;
    int h = blockHeight + 1;
    w /= 2;
    mpResManager->requestTextureResource("tmp_data", ResourceFormat::R32Float, ResourceManager::kDefaultFlags, 1024, w * h * 13);//change texture size if change blocksize and features num
    mpResManager->requestTextureResource("out_data", ResourceFormat::R32Float, ResourceManager::kDefaultFlags, 1024, w * h * 13);//change texture size if change blocksize and features num


    //UnorderedAccessView::SharedPtr uavView = UnorderedAccessView::create(1,0,);

    // Create our graphics state and accumulation shader
    mpGfxState = GraphicsState::create();

    // mpDenoiseShader = FullscreenLaunch::create(kDenoiseFragShader);
    mpPreprocessShader = FullscreenLaunch::create(kAccumNoisyDataShader);
    mpRegression = ComputeProgram::createFromFile("regressionCP.hlsl", "fit");
    mpPostShader = FullscreenLaunch::create(kAccumFilteredDataShader);
    mpCPState = ComputeState::create();
    mpCPState->setProgram(mpRegression);
    mpRegressionVars = ComputeVars::create(mpRegression->getReflector());

    ConstantBuffer::SharedPtr pCB = ConstantBuffer::create((Program::SharedPtr)mpRegression, "PerFrameCB", 128/* 4 * 32 */);
    mpRegressionVars->setConstantBuffer("PerFrameCB", pCB);
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

	mNeedFboClear = true;
	mAccumCount = 0;
}


void BlockwiseMultiOrderFeatureRegression::clearFbos(RenderContext* pCtx)
{
	// Clear our FBOs
	// TODO
	mNeedFboClear = false;
}

void BlockwiseMultiOrderFeatureRegression::renderGui(Gui* pGui)
{
	int dirty = 0;
	dirty |= (int)pGui->addCheckBox(mDoDenoise ? "Do BMFR Denoise" : "Ignore the denoise stage", mDoDenoise);
	dirty |= (int)pGui->addCheckBox(mBMFR_preprocess ? "Do Pre-Process" : "Skip Pre-process", mBMFR_preprocess);
	dirty |= (int)pGui->addCheckBox(mBMFR_postprocess ? "Do Post-Process" : "Skip Post-process", mBMFR_postprocess);
	dirty |= (int)pGui->addCheckBox(mBMFR_regression ? "Do Regression" : "Skip Regression", mBMFR_regression);
	dirty |= (int)pGui->addCheckBox(mBMFR_addNoise ? "Add Noise" : "Ignore features", mBMFR_addNoise);

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

	mInputTex.curNoisy = mpResManager->getTexture(mDenoiseChannel);
	mInputTex.curPos = mpResManager->getTexture("WorldPosition");
	mInputTex.curNorm = mpResManager->getTexture("WorldNormal");

	mInputTex.prevPos = mpResManager->getTexture("BMFR_PrevPos");
	mInputTex.prevNorm = mpResManager->getTexture("BMFR_PrevNorm");
	mInputTex.prevNoisy = mpResManager->getTexture("BMFR_PrevNoisy");
	mInputTex.prevFiltered = mpResManager->getTexture("BMFR_PrevFiltered");
	mInputTex.tmp_data = mpResManager->getTexture("tmp_data");
	mInputTex.out_data = mpResManager->getTexture("out_data");

	mInputTex.accept_bools = mpResManager->getTexture("BMFR_AcceptedBools");
	mInputTex.prevFramePixel = mpResManager->getTexture("BMFR_PrevFramePixel");

	mInputTex.output = mpResManager->getTexture("BMFR_Output");

	////pass four variables in, world pos, world normal, diffuse color and curr frame image
	//denoiseShaderVars["gPos"] = mpResManager->getTexture("WorldPosition");
	//denoiseShaderVars["gNorm"] = mpResManager->getTexture("WorldNormal");
	//denoiseShaderVars["gDiffuseMatl"] = mpResManager->getTexture("MaterialDiffuse");

	// Peform BMFR
	if (mBMFR_preprocess) {
		accumulate_noisy_data(pRenderContext);
	}
	
	pRenderContext->blit(mInputTex.curNoisy->getSRV(), mInputTex.prevNoisy->getRTV());
	pRenderContext->blit(mInputTex.curNorm->getSRV(), mInputTex.prevNorm->getRTV());
	pRenderContext->blit(mInputTex.curPos->getSRV(), mInputTex.prevPos->getRTV());

	if (mBMFR_regression) {
		fit_noisy_color(pRenderContext);
	}
	
	if (mBMFR_postprocess) {
		accumulate_filtered_data(pRenderContext);

		pRenderContext->blit(mInputTex.output->getSRV(), mInputTex.curNoisy->getRTV());
		pRenderContext->blit(mInputTex.output->getSRV(), mInputTex.prevFiltered->getRTV());
	}
	
	mAccumCount++;
}

void BlockwiseMultiOrderFeatureRegression::accumulate_noisy_data(RenderContext* pRenderContext)
{
	mpPreprocessShader->setCamera(mpScene->getActiveCamera());

	// Setup textures for our accumulate_noisy_data shader pass
	auto mpPreprocessShaderVars = mpPreprocessShader->getVars();
	mpPreprocessShaderVars["gCurPos"] = mInputTex.curPos;
	mpPreprocessShaderVars["gCurNorm"] = mInputTex.curNorm;
	mpPreprocessShaderVars["gCurNoisy"] = mInputTex.curNoisy;

	mpPreprocessShaderVars["gPrevPos"] = mInputTex.prevPos;
	mpPreprocessShaderVars["gPrevNorm"] = mInputTex.prevNorm;
	mpPreprocessShaderVars["gPrevNoisy"] = mInputTex.prevNoisy;

	mpPreprocessShaderVars["accept_bools"] = mInputTex.accept_bools;
	mpPreprocessShaderVars["out_prev_frame_pixel"] = mInputTex.prevFramePixel;

	// Setup variables for our accumulate_noisy_data pass
	mpPreprocessShaderVars["PerFrameCB"]["frame_number"] = mAccumCount;
	mpPreprocessShaderVars["PerFrameCB"]["IMAGE_WIDTH"] = mInputTex.curNoisy->getWidth();
	mpPreprocessShaderVars["PerFrameCB"]["IMAGE_HEIGHT"] = mInputTex.curNoisy->getHeight();

	// Execute the accumulate_noisy_data pass
	mpPreprocessShader->execute(pRenderContext, mpGfxState);
}


void BlockwiseMultiOrderFeatureRegression::accumulate_filtered_data(RenderContext* pRenderContext)
{
	auto mpPostVars = mpPostShader->getVars();
	mpPostVars["filtered_frame"] = mInputTex.curNoisy; // TODO, change name

	mpPostVars["accumulated_prev_frame"] = mInputTex.prevFiltered;

	mpPostVars["albedo"] = mpResManager->getTexture("MaterialDiffuse");
	mpPostVars["in_prev_frame_pixel"] = mInputTex.prevFramePixel;
	mpPostVars["accept_bools"] = mInputTex.accept_bools;

	mpPostVars["PerFrameCB"]["frame_number"] = mAccumCount;

	mpPostVars["accumulated_frame"] = mInputTex.output;

	mpPostShader->execute(pRenderContext, mpGfxState);

}

void BlockwiseMultiOrderFeatureRegression::fit_noisy_color(RenderContext* pRenderContext) {
	mpRegressionVars->setTexture("gCurPos", mInputTex.curPos);
	mpRegressionVars->setTexture("gCurNorm", mInputTex.curNorm);
	mpRegressionVars->setTexture("tmp_data", mInputTex.tmp_data);
	mpRegressionVars->setTexture("out_data", mInputTex.out_data);
	mpRegressionVars->setTexture("gCurNoisy", mInputTex.curNoisy);
	mpRegressionVars->setTexture("albedo", mpResManager->getTexture("MaterialDiffuse"));

	if (!mBMFR_addNoise) {
		mpRegression->addDefine("IGNORE_LD_fEATURES");
	}
	else {
		mpRegression->removeDefine("IGNORE_LD_fEATURES");
	}

	// Setup constant buffer
	ConstantBuffer::SharedPtr pcb = mpRegressionVars->getConstantBuffer("PerFrameCB");
	cbData[0] = mAccumCount;
	int width = mInputTex.curNoisy->getWidth();
	cbData[1] = width;
	int height = mInputTex.curNoisy->getHeight();
	cbData[2] = height;
	width = (width + 31) / 32; // 32 is block edge length
	height = (height + 31) / 32;
	int w = width + 1;
	int h = height + 1;

	w /= 2;
	cbData[3] = w;

	pcb->setBlob(cbData, 0, 128/*4 * 32*/);
	
	pRenderContext->pushComputeState(mpCPState);
	pRenderContext->pushComputeVars(mpRegressionVars);
	pRenderContext->dispatch(w * h, 1, 1);
	pRenderContext->popComputeVars();
	pRenderContext->popComputeState();
}