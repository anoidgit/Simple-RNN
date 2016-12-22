--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard SoftMax activation function,
	It forward the sequence from head to tail,
	and backward in reverse order.

	Designed for Recurrent Neural Networks

	Version 0.0.1

]]

local aSoftMax = torch.class('nn.aSoftMax', 'nn.Module')

function aSoftMax:__init(transpose)

	self.transpose = transpose

	self:clearState()

end

-- evaluate
function aSoftMax:evaluate()

	self.train = false

	self:clearState()

end

-- train
function aSoftMax:training()

	self.train = true

	self:clearState()

end

function aSoftMax:backward(input, gradOutput, scale)

	return self:updateGradInput(input, gradOutput)

end

function aSoftMax:updateOutput(input)

	self.gradInput = nil

	local output = input.new()
	output:resizeAs(input)

	if self.transpose and input:nDimension()==2 then

		input.THNN.SoftMax_updateOutput(
			input:t():cdata(),
			output:cdata()
		)

		output = output:t()

	else

		input.THNN.SoftMax_updateOutput(
			input:cdata(),
			output:cdata()
		)

	end

	self.output = output

	return self.output

end

function aSoftMax:updateGradInput(input, gradOutput)

	local gradInput = input.new()
	gradInput:resizeAs(input)

	if self.transpose and input:nDimension()==2 then

		input.THNN.SoftMax_updateGradInput(
			input:t():cdata(),
			gradOutput:t():cdata(),
			gradInput:cdata(),
			self.output:t():cdata()
		)

		gradInput = gradInput:t()

	else

		input.THNN.SoftMax_updateGradInput(
			input:cdata(),
			gradOutput:cdata(),
			gradInput:cdata(),
			self.output:cdata()
		)

	end

	self.gradInput = gradInput
	self.output = nil

	return self.gradInput

end