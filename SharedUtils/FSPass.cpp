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

#include "FSPass.h"

using namespace Falcor;

FSPass::SharedPtr FSPass::FSPass::create(const char *fragShader)
{
	return SharedPtr(new FSPass(fragShader));
}

FSPass::FSPass(const char *fragShader)
{ 
	mpPass = FullScreenPass::create(fragShader);
	mpVars = GraphicsVars::create( mpPass->getProgram()->getActiveVersion()->getReflector() );
}

void FSPass::updateActiveShader()
{
	mpVars = GraphicsVars::create(mpPass->getProgram()->getActiveVersion()->getReflector());
}

bool FSPass::setTexture(const std::string& name, const Texture::SharedPtr& pTexture)
{
	if (!isVarValid(name, ReflectionResourceType::Type::Texture)) return false;

	return mpVars->setTexture(name, pTexture);
}

void FSPass::execute(RenderContext::SharedPtr pRenderContext, GraphicsState::SharedPtr pGfxState)
{
	pRenderContext->pushGraphicsState(pGfxState);
	pRenderContext->pushGraphicsVars(mpVars);
		mpPass->execute(pRenderContext.get());
	pRenderContext->popGraphicsVars();
	pRenderContext->popGraphicsState();
}

void FSPass::setCamera(Falcor::Camera::SharedPtr pActiveCamera)
{
	// Shouldn't need to change unless Falcor internals do
	const char*__internalCB = "InternalPerFrameCB";
	const char*__internalVarName = "gCamera";

	// Actually set the internals
	ConstantBuffer::SharedPtr perFrameCB = mpVars[__internalCB];
	if (perFrameCB)
	{
		pActiveCamera->setIntoConstantBuffer(perFrameCB.get(), __internalVarName);
	}
}

void FSPass::setLights(const std::vector< Falcor::Light::SharedPtr > &pLights)
{
	// Shouldn't need to change unless Falcor internals do
	const char*__internalCB = "InternalPerFrameCB";
	const char*__internalCountName = "gLightsCount";
	const char*__internalLightsName = "gLights";

	// Actually set the internals
	ConstantBuffer::SharedPtr perFrameCB = mpVars[__internalCB];
	if (perFrameCB)
	{
		perFrameCB[__internalCountName] = uint32_t(pLights.size());
		const auto& pLightOffset = perFrameCB->getBufferReflector()->findMember(__internalLightsName);
		size_t lightOffset = pLightOffset ? pLightOffset->getOffset() : ConstantBuffer::kInvalidOffset;
		for (uint32_t i = 0; i < uint32_t(pLights.size()); i++)
		{
			pLights[i]->setIntoProgramVars(mpVars.get(), perFrameCB.get(), i * Light::getShaderStructSize() + lightOffset);
		}
	}
}

void FSPass::SharedPtr::Idx1::operator=(const Falcor::Texture::SharedPtr& pTexture) 
{ 
	// Set the texture
	bool wasSet = mpBuf->setTexture(mVar, pTexture);

	// If you triggered this assert, you:
	//   a) have called something like: myFSPass["someName"] = myTexture;
	//   b) and either variable "someName" does not exist in your shader, 
	//   c) *or* variable "someName" has a type that is *not* a texture.
	//
	// Commenting out this assert is fine if you do not mind that this assignment operator will do nothing.
	assert(wasSet);
}

bool FSPass::isVarValid(const std::string &varName, ReflectionResourceType::Type varType)
{
	ReflectionVar::SharedConstPtr mRes = mpVars->getReflection()->getResource(varName);
	if (mRes.get())
	{
		const ReflectionResourceType* pType = mRes->getType()->unwrapArray()->asResourceType();
		if (pType && pType->getType() == varType)
		{
			return true;
		}
	}
	return false;
}
