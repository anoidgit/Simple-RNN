--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard SoftMax activation function,
	It forward the sequence from head to tail,
	and backward in reverse order.

	Designed for Recurrent Neural Networks

	Version 0.0.1

]]

local aSeqSoftMax = torch.class('nn.aSeqSoftMax', 'nn.Module')

function aSeqSoftMax:__init(reverseOrder, transpose)

	if reverseOrder then
		self.rindex = nil
	else
		self.rindex = 1
	end

	self.transpose = transpose

	self:clearState()

end

-- evaluate
function aSeqSoftMax:evaluate()

	self.train = false

	self:clearState()

end

-- train
function aSeqSoftMax:training()

	self.train = true

	self:clearState()

end

function aSeqSoftMax:backward(input, gradOutput, scale)

	return self:updateGradInput(input, gradOutput)

end

function aSeqSoftMax:updateOutput(input)

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

	if self.train then
		table.insert(self._output, output)
	end

	self.output = output

	return self.output

end

function aSeqSoftMax:updateGradInput(input, gradOutput)

	local output = table.remove(self._output, self.rindex)
	local gradInput = input.new()
	gradInput:resizeAs(input)

	if self.transpose and input:nDimension()==2 then

		input.THNN.SoftMax_updateGradInput(
			input:t():cdata(),
			gradOutput:t():cdata(),
			gradInput:cdata(),
			output:t():cdata()
		)

		gradInput = gradInput:t()

	else

		input.THNN.SoftMax_updateGradInput(
			input:cdata(),
			gradOutput:cdata(),
			gradInput:cdata(),
			output:cdata()
		)

	end

	self.gradInput = gradInput

	return self.gradInput

end

-- Warning: This method is dangerous,
-- unless you know what you are doing.
function aSeqSoftMax:clearState()

	self._output = {}

end

function aSeqSoftMax:forget()

	self:clearState()

end
