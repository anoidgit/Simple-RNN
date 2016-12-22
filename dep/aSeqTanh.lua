--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard tanh activation function,
	It forward the sequence from head to tail,
	and backward in reverse order.

	Designed for Recurrent Neural Networks

	Version 0.1.0

]]

local aSeqTanh = torch.class('nn.aSeqTanh', 'nn.Module')

function aSeqTanh:__init(reverseOrder)

	if reverseOrder then
		self.rindex = nil
	else
		self.rindex = 1
	end

	self:clearState()

end

-- evaluate
function aSeqTanh:evaluate()

	self.train = false

	self:clearState()

end

-- train
function aSeqTanh:training()

	self.train = true

	self:clearState()

end

function aSeqTanh:backward(input, gradOutput, scale)

	return self:updateGradInput(input, gradOutput)

end

function aSeqTanh:updateOutput(input)

	self.gradInput = nil

	local output = input.new()
	output:resizeAs(input)

	input.THNN.Tanh_updateOutput(
		input:cdata(),
		output:cdata()
	)
	if self.train then
		table.insert(self._output,output)
	end
	self.output = output

	return self.output

end

function aSeqTanh:updateGradInput(input, gradOutput)

	self.output = nil

	local output = table.remove(self._output, self.rindex)

	self.gradInput = self.gradInput or input.new()
	self.gradInput:resizeAs(input)
	input.THNN.Tanh_updateGradInput(
		input:cdata(),
		gradOutput:cdata(),
		self.gradInput:cdata(),
		output:cdata()
	)
	return self.gradInput

end

-- Warning: This method is dangerous,
-- unless you know what you are doing.
function aSeqTanh:clearState()

	self._output = {}

end

function aSeqTanh:forget()

	self:clearState()

end
