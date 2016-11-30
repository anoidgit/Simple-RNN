--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard tanh activation function

	Version 0.0.1

]]

local aSeqTanh = torch.class('nn.aSeqTanh', 'nn.Module')

function aSeqTanh:__init()

	self.backward = self.updateGradInput
	self:clearState()

end

function aSeqTanh:updateOutput(input)

	local output = torch.tanh(input)
	table.insert(self._output,output)
	self.output = output

	return self.output

end

function aSeqTanh:updateGradInput(input, gradOutput)

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
function aSeqTanh:clearState()
	self._output = {}
end
