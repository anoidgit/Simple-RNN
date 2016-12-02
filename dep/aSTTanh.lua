--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard tanh activation function

	Version 0.0.4

]]

local aSTTanh = torch.class('nn.aSTTanh', 'nn.Module')

function aSTTanh:backward(input, gradOutput, scale)

	return self:updateGradInput(input, gradOutput)

end

function aSTTanh:updateOutput(input)

	self.output = torch.tanh(input)

	return self.output

end

function aSTTanh:updateGradInput(input, gradOutput)

	local output = torch.tanh(input)
	local gradInput = input.new()
	gradInput:resizeAs(input):fill(1)
	gradInput:addcmul(-1,output,output)
	gradInput:cmul(gradOutput)
	self.gradInput = gradInput

	return self.gradInput

end
