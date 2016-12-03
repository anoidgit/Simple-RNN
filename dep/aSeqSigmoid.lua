--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard sigmoid activation function,
	It forward the sequence from head to tail,
	and backward in reverse order.

	Designed for Recurrent Neural Networks

	Version 0.0.4

]]

local aSeqSigmoid = torch.class('nn.aSeqSigmoid', 'nn.Module')

function aSeqSigmoid:__init(reverseOrder)

	if reverseOrder then
		self.rindex = nil
	else
		self.rindex = 1
	end

	self:forget()

end

-- evaluate
function aSeqSigmoid:evaluate()

	self.train = false

	self:forget()

end

-- train
function aSeqSigmoid:training()

	self.train = true

	self:forget()

end

function aSeqSigmoid:backward(input, gradOutput, scale)

	return self:updateGradInput(input, gradOutput)

end

function aSeqSigmoid:updateOutput(input)

	local output = torch.sigmoid(input)
	if self.train then
		table.insert(self._output,output)
	end
	self.output = output

	return self.output

end

function aSeqSigmoid:updateGradInput(input, gradOutput)

	local output = table.remove(self._output, self.rindex)

	local gradInput = input.new()
	gradInput:resizeAs(input):fill(1)
	gradInput:csub(output)
	gradInput:cmul(output)
	gradInput:cmul(gradOutput)
	self.gradInput = gradInput

	return self.gradInput

end

-- Warning: This method is dangerous,
-- unless you know what you are doing.
function aSeqSigmoid:forget()

	self._output = {}

end
