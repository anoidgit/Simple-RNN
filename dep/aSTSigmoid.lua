--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard sigmoid activation function

	Version 0.0.2

]]

local aSTSigmoid = torch.class('nn.aSTSigmoid', 'nn.Module')

function aSTSigmoid:backward(input, gradOutput, scale)

	return self:updateGradInput(input, gradOutput)

end

function aSTSigmoid:updateOutput(input)

	self.output = torch.sigmoid(input)

	return self.output

end

function aSTSigmoid:updateGradInput(input, gradOutput)

	local output = torch.sigmoid(input)
	local gradInput = input.new()
	gradInput:resizeAs(input):fill(1)
	gradInput:csub(output)
	gradInput:cmul(output)
	gradInput:cmul(gradOutput)
	self.gradInput = gradInput

	return self.gradInput

end
