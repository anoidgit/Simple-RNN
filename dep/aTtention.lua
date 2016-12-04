--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a softmax soft attention function

	Designed for Recurrent Neural Networks

	Version 0.0.2

]]

local aTtention = torch.class('nn.aTtention', 'nn.Module')

function aTtention:__init(vecsize, reverseOrder)

	self.vecsize = vecsize

	if reverseOrder then
		self.rindex = nil
	else
		self.rindex = 1
	end

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

	local src = input[1]--seqlen(*batchsize)*vecsize
	local _iSize = src:size()
	local ndim = #_iSize
	local seqlen = _iSize[1]
	local batchsize
	_iSize[1] = 1
	gradOutput = gradOutput:reshape(_iSize)--1(*batchsize)*vecsize
	_iSize[1] = seqlen
	gradOutput = gradOutput:expand(_iSize)-- seqlen(*batchsize)*vecsize
	local gradScore = torch.cmul(gradOutput, src):sum(ndim)
	if ndim == 3 then
		batchsize = _iSize[2]
		gradScore = gradScore:reshape(seqlen,batchsize)
	else
		gradScore = gradScore:reshape(seqlen)
	end
	self.gradInput = self.module:backward(input, gradScore, scale)-- {seqlen(*batchsize)*vecsize, (batchsize*)vecsize}
	local score = table.remove(self._score, self.rindex)-- seqlen(*batchsize)*vecsize
	self.gradInput[1]:add(torch.cmul(gradOutput, score))

	return self.gradInput

end

function aTtention:updateOutput(input)

	local score = self.module:updateOutput(input)-- seqlen(*batchsize)
	local src = input[1]
	local _iSize = src:size()
	local ndim = #_iSize
	_iSize[ndim] = 1
	score = score:reshape(_iSize)-- seqlen(*batchsize)*1
	_iSize[ndim] = self.vecsize
	score = score:expand(_iSize)-- seqlen(*batchsize)*vecsize
	if self.train then
		table.insert(self._score, score)
	end
	self.output = torch.cmul(score, src):sum(1)-- 1(*batchsize)*vecsize
	if ndim == 3 then
		batchsize = _iSize[2]
		self.output:reshape(batchsize, self.vecsize)-- batchsize*vecsize
	else
		self.output:reshape(self.vecsize) 
	end

	return self.output

end

function aTtention:updateGradInput(input, gradOutput)
	return self:backward(input, gradOutput)
end

function aTtention:prepare()

	require "aSeqBiLinear"
	nn.aSequential = nn.Sequential
	nn.aSoftMax = nn.SoftMax
	nn.aTranspose = nn.Transpose

end

function aTtention:reset()

	self.module = nn.Sequential()
		:add(nn.aSeqBiLinear(self.vecsize, self.rindex==nil))
		:add(nn.aTranspose({1,2}))
		:add(nn.aSoftMax())
		:add(nn.aTranspose({1,2}))

	self.modules = {self.module}

end

-- Warning: This method is dangerous,
-- unless you know what you are doing.
function aTtention:clearState()

	self._score = {}

end

function aTtention:forget()

	self:clearState()

end
