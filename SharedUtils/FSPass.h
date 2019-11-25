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

#pragma once

#include "Falcor.h"

/** This is a very light wrapper around Falcor::FullScreenPass that removes some of the boilerplate of
calling and initializing a FullScreenPass pass and enables access to constant buffers and textures
using an array [] notation (with a slight bit more error checking that the current Falcor API).

Initialization:
   FSPass::SharedPtr mpMyPass = FSPass::create("myFullScreenShader.ps.hlsl");

Pass setup:
	mpMyPass["myShaderCB"]["myVar"] = uint4( 1, 2, 4, 16 );
	mpMyPass["myShaderCB"]["myBinaryBlob"].setBlob( cpuSideBinaryBlob );
	mpMyPass["myShaderTexture"] = myTextureResource;

Pass execution:
    pGraphicsState->setFbo( pOutputFbo );  
	mpMyPass->execute( pRenderContext, pGraphicsState );

*/
class FSPass : public std::enable_shared_from_this<FSPass>
{
public:

	// This class uses an overloaded shared_ptr that allows calling array operations.
	class SharedPtr : public std::shared_ptr<FSPass>
	{
	public:

		// A secondary intermediary class that allows a double [][] operator to be used on the SharedPtr.
		class Var
		{
		public:
			// Constructor gets called when mySharedPtr["myIdx1"]["myVar"] is encountered
			Var(Falcor::ConstantBuffer *cb, const std::string& var) : mCB(cb), mVar(var) { if (cb) { mOffset = cb->getVariableOffset(var); } }

			// Assignment operator gets called when mySharedPtr["myIdx1"]["myVar"] = T(someData); is encountered
			template<typename T> void operator=(const T& val) { if (mOffset != Falcor::VariablesBuffer::kInvalidOffset) { mCB->setVariable(mOffset, val); } }

			// Allows mySharedPtr["myIdx1"]["myVar"].setBlob( blobData )...    In theory, block binary transfers could be done with operator=, but that could accidentally do implicit binary transfers
			template<typename T> void setBlob(const T& blob) { if (mOffset != Falcor::VariablesBuffer::kInvalidOffset) { mCB->setBlob(&blob, mOffset, sizeof(T)); } }
		protected:
			Falcor::ConstantBuffer *mCB;
			const std::string mVar;
			size_t mOffset = Falcor::VariablesBuffer::kInvalidOffset;
		};

		// An intermediary class that allows the [] operator to be used on the SharedPtr.
		class Idx1
		{
		public:
			// Constructor gets called when mySharedPtr["myIdx1"] is encoutered
			Idx1(FSPass* pBuf, const std::string& var) : mpBuf(pBuf), mVar(var) { }

			// When a second array operator is encountered, instatiate a Var object to handle mySharedPtr["myIdx1"]["myVar"]
			Var operator[](const std::string& var) { return Var(mpBuf->mpVars->getConstantBuffer(mVar).get(), var); }

			// When encountering an assignment operator of a texture, treat mySharedPtr["myIdx1"] = pTexture; as a call to
			//     mpVars->setTexture( "myIdx1", pTexture );
			void operator=(const Falcor::Texture::SharedPtr& pTexture);

			// Allow conversion of this intermediary type to a constant buffer, e.g., for ConstantBuffer::SharedPtr cb = mySharedPtr["myIdx1"];
			operator Falcor::ConstantBuffer::SharedPtr() { return mpBuf->mpVars->getConstantBuffer(mVar); }

		protected:
			FSPass* mpBuf;
			const std::string mVar;
		};

		SharedPtr() = default;
		SharedPtr(FSPass* pBuf) : std::shared_ptr<FSPass>(pBuf) {}
		SharedPtr(std::shared_ptr<FSPass> pBuf) : std::shared_ptr<FSPass>(pBuf) {}

		// Calling [] on the SharedPtr?  Create an intermediate object to process further operators
		Idx1 operator[](const std::string& var) { return Idx1(get(), var); }
	};

	// public constructor
	static SharedPtr create(const char *fragShader);
	virtual ~FSPass() = default;

	// Execute the full-screen shader
	void execute(Falcor::RenderContext::SharedPtr pRenderContext, Falcor::GraphicsState::SharedPtr pGfxState);

	// Set a variable
	template<typename T>
	void setVariable(const std::string& cBuf, const std::string& name, const T& value)
	{
		Falcor::ConstantBuffer::SharedPtr cb = mpVars->getConstantBuffer(cBuf);
		if (cb)
		{
			cb->setVariable(name, value);
		}
	}

	// Falcor / Slang has internal state not automatically set when using full-screen passes.  These are
	// helpers to set this state, allowing full-screen passes to use the Falcor built-ins.
	void setCamera(Falcor::Camera::SharedPtr pActiveCamera);
	void setLights(const std::vector< Falcor::Light::SharedPtr > &pLights);

	// Set a texture
	bool setTexture(const std::string& name, const Falcor::Texture::SharedPtr& pTexture);

	// If shader settings / #define's have changed, call this method to ensure the class updates
	//    the shader to the current active version of the shader.
	void updateActiveShader();

	// Get the current program
	Falcor::Program::SharedPtr getProgram()
	{
		return mpPass->getProgram();
	}

	// Get the current variables
	Falcor::GraphicsVars::SharedPtr getVars()
	{	
		return mpVars;
	}

	// TODO:  It's possible we'll want more accessors here to simplify access to mpVars, but for 
	// fairly simplistic use cases, this is probably sufficient.

protected:
	FSPass(const char *fragShader);

private:
	Falcor::FullScreenPass::UniquePtr mpPass;
	Falcor::GraphicsVars::SharedPtr   mpVars;

	bool isVarValid(const std::string &varName, Falcor::ReflectionResourceType::Type varType);
};
