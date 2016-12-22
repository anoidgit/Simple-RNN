--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a transfer module to fit NCE

	Designed for Recurrent Neural Networks

	Version 0.0.1

]]

local aTransfer = torch.class('nn.aTransfer', 'nn.Module')

function aTransfer:__init(tranTo)

	self.tranTo = tranTo

end

function aTransfer:_recursiveFill(input, src, tar)

	local rs
	if torch.isTensor(input) then
		rs = input:clone()
		rs[rs:eq(src)] = tar
	else
		rs = {}
		for _, v in ipairs(input) do
			rs[_] = self:_recursiveFill(v, src, tar)
		end
	end
	return rs

end

function aTransfer:_recursiveZero(input)

	local rs
	if torch.isTensor(input) then
		rs = input:clone():zero()
	else
		rs = {}
		for _, v in ipairs(input) do
			rs[_] = self:_recursiveZero(v)
		end
	end
	return rs

end

function aTransfer:backward(input, gradOutput, scale)

	self.gradInput = self:_recursiveZero(gradOutput)

	return self.gradInput

end

function aTransfer:updateOutput(input)

	self.output = self:_recursiveFill(input, 0, self.tranTo)

	return self.output

end