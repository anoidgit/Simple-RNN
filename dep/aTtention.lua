--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a softmax soft attention function

	Designed for Recurrent Neural Networks

	Version 0.0.8

]]

local aTtention = torch.class('nn.aTtention', 'nn.Container')

function aTtention:__init(vecsize, maskzero)

	self.vecsize = vecsize

	self.maskzero = maskzero

	self:reset()

end

-- evaluate
function aTtention:evaluate()

	self.train = false

	self.module:evaluate()

	self:clearState()

end

-- train
function aTtention:training()

	self.train = true

	self.module:training()

	self:clearState()

end

function aTtention:backward(input, gradOutput, scale)

	self.output = nil
	local src = input[1]--seqlen(*batchsize)*vecsize
	if self.maskzero then
		self:_maskZero(input[2], gradOutput)
	end
	local _iSize = src:size()
	local _sdim = #_iSize - 1
	local seqlen = _iSize[1]

	-- gradOutput (batchsize*)vecsize
	-- gradScore seqlen(*batchsize)
	local gradScore = src.new()
	if _sdim == 2 then
		gradScore:resize(seqlen, _iSize[2])
	else
		gradScore:resize(seqlen)
	end
	for _ = 1, seqlen do
		gradScore[_]:copy(torch.cmul(src[_], gradOutput):sum(_sdim))
	end

	self.gradInput = self.module:backward(input, gradScore, scale)-- {seqlen(*batchsize)*vecsize, (batchsize*)vecsize}
	local _srcsize = src[1]:size()
	local _rsize = src[1]:size()
	_rsize[_sdim] = 1
	local gradSrc = self.gradInput[1]
	for _ = 1, seqlen do
		gradSrc[_]:add(torch.cmul(gradOutput, self.score[_]:reshape(_rsize):expand(_srcsize)))
	end
	self.score = nil

	return self.gradInput

end

function aTtention:_maskZero(std, score)

	local std_zero = std[1]:clone():zero()
	for _ = 1, std:size(1) do
		if std[_]:equal(std_zero) then
			score[_]:zero()
		end
	end

end

function aTtention:updateOutput(input)

	self.gradInput = nil
	self.output = nil
	local score = self.module:updateOutput(input)-- seqlen(*batchsize)

	local src = input[1]
	local _iSize = src:size()
	local _sdim = #_iSize - 1
	local seqlen = _iSize[1]

	if self.train then
		self.score = score
	end

	local output
	local _srcsize = src[1]:size()
	local _rsize = src[1]:size()
	_rsize[_sdim] = 1
	for _ = 1, seqlen do
		if _ == 1 then
			output = torch.cmul(src[_], score[_]:reshape(_rsize):expand(_srcsize))
		else
			output:add(torch.cmul(src[_], score[_]:reshape(_rsize):expand(_srcsize)))
		end
	end

	if self.maskzero then
		self:_maskZero(input[2], output)
	end

	self.output = output

	return self.output

end

function aTtention:updateGradInput(input, gradOutput)
	return self:backward(input, gradOutput)
end

function aTtention:reset()

	self.module = nn.Sequential()
		:add(nn.aBiLinearScore(self.vecsize))
		--:add(nn.aTranspose({1,2}))
		:add(nn.aTSoftMax(true))
		--:add(nn.aTranspose({1,2}))

	self.modules = {self.module}

end

-- Warning: This method is dangerous,
-- unless you know what you are doing.
function aTtention:clearState()

	self.score = nil

end

function aTtention:forget()

	self:clearState()

end
