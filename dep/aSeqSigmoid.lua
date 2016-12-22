--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard sigmoid activation function,
	It forward the sequence from head to tail,
	and backward in reverse order.

	Designed for Recurrent Neural Networks

	Version 0.1.0

]]

local aSeqSigmoid = torch.class('nn.aSeqSigmoid', 'nn.Module')

function aSeqSigmoid:__init(reverseOrder)

	if reverseOrder then
		self.rindex = nil
	else
		self.rindex = 1
	end

	self:clearState()

end

-- evaluate
function aSeqSigmoid:evaluate()

	self.train = false

	self:clearState()

end

-- train
function aSeqSigmoid:training()

	self.train = true

	self:clearState()

end

function aSeqSigmoid:backward(input, gradOutput, scale)

	return self:updateGradInput(input, gradOutput)

end

function aSeqSigmoid:updateOutput(input)

	self.gradInput = nil

	local output = input.new()
	output:resizeAs(input)
	input.THNN.Sigmoid_updateOutput(
		input:cdata(),
		output:cdata()
	)
	if self.train then
		table.insert(self._output,output)
	end
	self.output = output

	return self.output

end

function aSeqSigmoid:updateGradInput(input, gradOutput)

	self.output = nil

	local output = table.remove(self._output, self.rindex)

	self.gradInput = self.gradInput or input.new()
	input.THNN.Sigmoid_updateGradInput(
		input:cdata(),
		gradOutput:cdata(),
		self.gradInput:cdata(),
		output:cdata()
	)

	return self.gradInput

end

-- Warning: This method is dangerous,
-- unless you know what you are doing.
function aSeqSigmoid:clearState()

	self._output = {}

end

function aSeqSigmoid:forget()

	self:clearState()

end
