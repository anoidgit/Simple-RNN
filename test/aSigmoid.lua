--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard sigmoid activation function

	Version 0.0.1

]]

local aSigmoid = torch.class('nn.aSigmoid', 'nn.Module')

function aSigmoid:__init()

	self.backward = self.updateGradInput

end

function aSigmoid:updateOutput(input)

	self.output = torch.sigmoid(input)

	return self.output

end

function aSigmoid:updateGradInput(input, gradOutput)

	local gradInput = input.new()
	gradInput:resizeAs(input):fill(1)
	gradInput:csub(self.output)
	gradInput:cmul(self.output)
	gradInput:cmul(gradOutput)
	self.gradInput = gradInput

	return self.gradInput

end
