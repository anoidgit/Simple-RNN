--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a Repeat Module

	Version 0.0.2

]]

local aRepeat, parent = torch.class('nn.aRepeat', 'nn.Module')

-- generate a module
function aRepeat:__init(times, dim, ndim)

	self.times = times

	self.dim = dim or 1
	self.ndim = ndim or self.dim

end

function aRepeat:updateOutput(input)

	local iSize = input:size()
	local idim = #iSize
	local dim = self.dim + idim - self.ndim
	for _ = 1, idim do
		iSize[_] = 1
	end
	iSize[dim] = self.times
	self.output = input:repeatTensor(iSize)

	return self.output

end

function aRepeat:updateGradInput(input, gradOutput)

	local iSize = input:size()
	local idim = #iSize
	local dim = self.dim + idim - self.ndim
	local clengh = iSize[dim]
	local stid = 1
	self.gradInput = gradOutput:narrow(dim, 1, clengh):clone()
	for _ = 1, self.times - 1 do
		stid = stid + clengh
		self.gradInput:add(gradOutput:narrow(dim, stid, clengh))
	end

	return self.gradInput

end
