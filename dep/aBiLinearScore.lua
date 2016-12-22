--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a core for standard Bilinear
	Input should be {seqlen(*batchsize)*vecsize, (batchsize*)vecsize}

	This module was designed for aTtension.

	Version 0.0.7

]]

local aBiLinearScore = torch.class('nn.aBiLinearScore', 'nn.Container')

function aBiLinearScore:__init(vecsize)

	self.vecsize = vecsize

	self:reset()

end

function aBiLinearScore:training()

	self.train = true
	self.module:training()
	self:clearState()

end

function aBiLinearScore:evaluate()

	self.train = false
	self.module:evaluate()
	self:clearState()

end

function aBiLinearScore:updateOutput(input)

	self.gradInput = nil
	self.output = nil

	local src = input[1]-- seqlen(*batchsize)*vecsize
	local _iSize = src:size()
	local _sdim = #_iSize - 1
	local seqlen = _iSize[1]

	local _transfer = self.module:forward(input[2])-- (batchsize*)vecsize

	if self.train then
		self._transfer = _transfer
	end

	local output = src.new()
	if _sdim == 2 then
		output = output:resize(seqlen, _iSize[2])-- seqlen*batchsize
	else
		output = output:resize(seqlen)--seqlen
	end

	for _ = 1, seqlen do
		output[_]:copy(torch.cmul(src[_], _transfer):sum(_sdim))-- seqlen(*batchsize)
	end
	self.output = output

	return self.output

end

function aBiLinearScore:backward(input, gradOutput, scale)

	self.output = nil

	local src = input[1]

	local _iSize = src:size()
	local _sdim = #_iSize - 1
	local seqlen = _iSize[1]
	local gradSrc = src.new()
	gradSrc:resizeAs(src)
	local gradTrans

	local _srcsize = src[1]:size()
	local _rsize = src[1]:size()
	_rsize[_sdim] = 1
	for _ = 1, seqlen do
		local _cgrad = gradOutput[_]:reshape(_rsize):expand(_srcsize)
		if _ == 1 then
			gradTrans = torch.cmul(src[_], _cgrad)
		else
			gradTrans:add(torch.cmul(src[_], _cgrad))
		end
		gradSrc[_]:copy(torch.cmul(self._transfer, _cgrad))
	end

	self._transfer = nil

	-- backward gradTrans to gradStd
	local gradStd = self.module:backward(input[2], gradTrans, scale)

	self.gradInput = {gradSrc, gradStd}

	return self.gradInput

end

function aBiLinearScore:updateGradInput(input, gradOutput)

	return self:backward(input, gradOutput)

end

function aBiLinearScore:clearState()

	self._transfer = nil

end

function aBiLinearScore:forget()

	self:clearState()

end

function aBiLinearScore:reset()

	self.module = nn.aLinear(self.vecsize, self.vecsize, false)-- nn.Linear without bias
	self.modules = {self.module}

	self._transfer = nil

end

function aBiLinearScore:prepare()

	nn.aLinear = nn.Linear

end
