--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard tanh activation function,
	It forward the sequence from head to tail,
	and backward in reverse order.

	Designed for Recurrent Neural Networks

	Version 0.0.3

]]

local aRSeqTanh = torch.class('nn.aRSeqTanh', 'nn.Module')

function aRSeqTanh:__init()

	self.backward = self.updateGradInput
	self:forget()

end

function aRSeqTanh:updateOutput(input)

	local output = torch.tanh(input)
	table.insert(self._output,output)
	self.output = output

	return self.output

end

function aRSeqTanh:updateGradInput(input, gradOutput)

	local output = table.remove(self._output)

	local gradInput = input.new()
	gradInput:resizeAs(input):fill(1)
	gradInput:addcmul(-1, output, output)
	gradInput:cmul(gradOutput)
	self.gradInput = gradInput

	return self.gradInput

end

-- Warning: This method is dangerous,
-- unless you know what you are doing.
function aRSeqTanh:forget()
	self._output = {}
end
