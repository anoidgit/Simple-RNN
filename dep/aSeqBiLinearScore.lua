--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a core for standard Bilinear
	Input should be {seqlen(*batchsize)*vecsize, (batchsize*)vecsize}

	This module was designed for aTtension.

	Version 0.0.5

]]

local aSeqBiLinearScore = torch.class('nn.aSeqBiLinearScore', 'nn.Container')

function aSeqBiLinearScore:__init(vecsize, reverseOrder)

	self.vecsize = vecsize

	if reverseOrder then
		self.rindex = nil
	else
		self.rindex = 1
	end

	self:reset()

end

function aSeqBiLinearScore:training()

	self.train = true
	self.module:training()
	self:clearState()

end

function aSeqBiLinearScore:evaluate()

	self.train = false
	self.module:evaluate()
	self:clearState()

end

function aSeqBiLinearScore:updateOutput(input)

	local src = input[1]-- seqlen(*batchsize)*vecsize
	local std = input[2]-- (batchsize)*vecsize
	local _iSize = src:size()
	local ndim = #_iSize
	local seqlen = _iSize[1]
	local batchsize

	local _transfer = self.module:forward(std)-- (batchsize*)vecsize

	_iSize[1] = 1
	_transfer = _transfer:reshape(_iSize)-- 1(*batchsize)*vecsize
	_iSize[1] = seqlen
	_transfer = _transfer:expand(_iSize)-- seqlen(*batchsize)*vecsize

	if self.train then
		table.insert(self._transfer, _transfer)-- seqlen(*batchsize)*vecsize
	end

	self.output = torch.cmul(src, _transfer):sum(ndim)-- seqlen(*batchsize)*1
	if ndim == 3 then
		batchsize = _iSize[2]
		self.output = self.output:reshape(seqlen, batchsize)-- seqlen*batchsize
	else
		self.output = self.output:reshape(seqlen)--seqlen
	end

	return self.output

end

function aSeqBiLinearScore:backward(input, gradOutput, scale)

	local src = input[1]
	local std = input[2]

	local _iSize = src:size()
	local ndim = #_iSize
	local seqlen = _iSize[1]
	local batchsize
	local gradSrc

	local _transfer = table.remove(self._transfer, self.rindex)-- -- seqlen(*batchsize)*vecsize

	_iSize[ndim] = 1
	gradOutput = gradOutput:reshape(_iSize)-- seqlen(*batchsize)*1
	_iSize[ndim] = self.vecsize
	gradOutput = gradOutput:expand(_iSize)-- seqlen(*batchsize)*vecsize

	local gradTrans = torch.cmul(gradOutput, src):sum(1)-- 1(*batchsize)*vecsize

	local gradSrc = torch.cmul(gradOutput, _transfer)-- seqlen(*batchsize)*vecsize

	local gradStd

	-- backward gradTrans to gradStd
	if ndim == 3 then
		batchsize = _iSize[2]
		gradTrans = gradTrans:reshape(batchsize, self.vecsize)
		gradStd = self.module:backward(std, gradTrans, scale)-- (seqlen*batchsize)*vecsize
	else
		gradTrans = gradTrans:reshape(self.vecsize)-- vecsize
		gradStd = self.module:backward(std, gradTrans, scale)-- seqlen*vecsize
	end

	self.gradInput = {gradSrc, gradStd}

	return self.gradInput

end

function aSeqBiLinearScore:updateGradInput(input, gradOutput)

	return self:backward(input, gradOutput)

end

function aSeqBiLinearScore:clearState()

	self._transfer = {}

end

function aSeqBiLinearScore:forget()

	self:clearState()

end

function aSeqBiLinearScore:reset()

	self.module = nn.aLinear(self.vecsize, self.vecsize, false)-- nn.Linear without bias
	self.modules = {self.module}

	self._transfer ={}

end

function aSeqBiLinearScore:prepare()

	nn.aLinear = nn.Linear

end
