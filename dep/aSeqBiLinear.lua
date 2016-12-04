--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a core for standard Bilinear
	Input should be {seqlen(*batchsize)*vecsize, (batchsize*)vecsize}

	This module was designed for aTtension.

	Version 0.0.3

]]

local aSeqBiLinear = torch.class('nn.aSeqBiLinear', 'nn.Container')

function aSeqBiLinear:__init(vecsize, reverseOrder)

	self.vecsize = vecsize

	if reverseOrder then
		self.rindex = nil
	else
		self.rindex = 1
	end

	self:reset()

end

function aSeqBiLinear:training()

	self.train = true
	self.module:training()
	self:clearState()

end

function aSeqBiLinear:evaluate()

	self.train = false
	self.module:evaluate()
	self:clearState()

end

function aSeqBiLinear:updateOutput(input)

	local src = input[1]-- seqlen(*batchsize)*vecsize
	local std = input[2]-- (batchsize)*vecsize
	local _iSize = src:size()
	local ndim = #_iSize
	local seqlen = _iSize[1]
	local batchsize

	if ndim == 3 then
		batchsize = _iSize[2]
		local _tranSize = seqlen * batchsize
		src = src:reshape(_tranSize, self.vecsize)-- (seqlen*batchsize)*vecsize
	end

	local _transfer = self.module:forward(src)-- (seqlen(*batchsize))*vecsize

	if batchsize then
		src = src:reshape(seqlen, batchsize, self.vecsize)-- restore
	end

	if self.train then
		table.insert(self._transfer, _transfer)-- (seqlen(*batchsize))*vecsize
	end

	_iSize[1] = 1
	std = std:reshape(_iSize)-- 1(*batchsize)*vecsize
	_iSize[1] = seqlen
	std = std:expand(_iSize)-- (eqlen(*batchsize)*vecsize

	self.output = torch.cmul(std, _transfer):sum(ndim)-- seqlen(*batchsize)*1
	if batchsize then
		self.output = self.output:reshape(seqlen, batchsize)-- seqlen*batchsize
	else
		self.output = self.output:reshape(seqlen)--seqlen
	end

	return self.output

end

function aSeqBiLinear:backward(input, gradOutput, scale)

	local src = input[1]
	local std = input[2]

	local _iSize = src:size()
	local ndim = #_iSize
	local seqlen = _iSize[1]
	local batchsize
	local gradSrc

	local _transfer = table.remove(self._transfer, self.rindex)-- (seqlen(*batchsize))*vecsize

	_iSize[ndim] = 1
	gradOutput = gradOutput:reshape(_iSize)-- seqlen(*batchsize)*1
	_iSize[ndim] = self.vecsize
	gradOutput = gradOutput:expand(_iSize)-- seqlen(*batchsize)*vecsize
	_iSize[1] = 1
	std = std:reshape(_iSize)
	_iSize[1] = seqlen
	std = std:expand(_iSize)-- seqlen(*batchsize)*vecsize

	local gradStd = torch.cmul(gradOutput, _transfer):sum(1)-- 1(*batchsize)*vecsize

	local gradTrans = torch.cmul(gradOutput, std)-- seqlen(*batchsize)*vecsize

	local gradSrc
	-- reshape gradTrans to (seqlen(*batchsize))*vecsize here
	if ndim == 3 then
		batchsize = _iSize[2]
		gradStd = gradStd:reshape(batchsize, self.vecsize)
		local _tranSize = seqlen * batchsize
		gradTrans = gradTrans:reshape(_tranSize, self.vecsize)-- (seqlen*batchsize)*vecsize
		src = src:reshape(_tranSize, self.vecsize)-- (seqlen*batchsize)*vecsize
		gradSrc = self.module:backward(src, gradTrans, scale)-- (seqlen*batchsize)*vecsize
		gradSrc = gradSrc:reshape(_iSize)-- seqlen*batchsize*vecsize
	else
		gradStd = gradStd:reshape(self.vecsize)-- vecsize
		gradSrc = self.module:backward(src, gradTrans, scale)-- seqlen*vecsize
	end

	self.gradInput = {gradSrc, gradStd}

	return self.gradInput

end

function aSeqBiLinear:updateGradInput(input, gradOutput)

	return self:backward(input, gradOutput)

end

function aSeqBiLinear:clearState()

	self._transfer = {}

end

function aSeqBiLinear:forget()

	self:clearState()

end

function aSeqBiLinear:reset()

	self.module = nn.aLinear(self.vecsize, self.vecsize, false)-- nn.Linear without bias
	self.modules = {self.module}

	self._transfer ={}

end

function aSeqBiLinear:prepare()

	nn.aLinear = nn.Linear

end
