--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard tanh activation function

	Version 0.0.2

]]

local aTanh = torch.class('nn.aTanh', 'nn.Module')

function aTanh:__init()

	self.backward = self.updateGradInput

end

function aTanh:updateOutput(input)

	self.output = torch.tanh(input)

	return self.output

end

function aTanh:updateGradInput(input, gradOutput)

	local gradInput = input.new()
	gradInput:resizeAs(input):fill(1)
	gradInput:addcmul(-1,gradOutput,gradOutput)
	self.gradInput = gradInput

	return gradInput

end
