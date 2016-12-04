--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a JoinTable

	Designed for Recurrent Neural Networks

	Version 0.0.2

]]

local aSeqJoinTable, parent = torch.class('nn.aSeqJoinTable', 'nn.Module')

function aSeqJoinTable:__init(dimension, nInputDims, reverseOrder)
	parent.__init(self)
	self.size = torch.LongStorage()
	self.dimension = dimension
	self.gradInput = {}
	self.nInputDims = nInputDims
	if reverseOrder then
		self.rindex = nil
	else
		self.rindex = 1
	end
	self:clearState()
end

function aSeqJoinTable:training()

	self.train = true
	self:clearState()

end

function aSeqJoinTable:evaluate()

	self.train = false
	self:clearState()

end

function aSeqJoinTable:clearState()

	self._output = {}

end

function aSeqJoinTable:forget()

	self:clearState()

end

function aSeqJoinTable:_getPositiveDimension(input)
	local dimension = self.dimension
	if dimension < 0 then
		dimension = input[1]:dim() + dimension + 1
	elseif self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
		dimension = dimension + 1
	end
	return dimension
end

function aSeqJoinTable:updateOutput(input)
	local dimension = self:_getPositiveDimension(input)

	for i=1,#input do
		local currentOutput = input[i]
		if i == 1 then
			self.size:resize(currentOutput:dim()):copy(currentOutput:size())
		else
			self.size[dimension] = self.size[dimension]
				+ currentOutput:size(dimension)
		end
	end
	local output = input[1].new()
	output:resize(self.size)

	local offset = 1
	for i=1,#input do
		local currentOutput = input[i]
		output:narrow(dimension, offset,
			currentOutput:size(dimension)):copy(currentOutput)
		offset = offset + currentOutput:size(dimension)
	end
	if self.train then
		table.insert(self._output, output)
	end
	self.output = output
	return self.output
end

function aSeqJoinTable:updateGradInput(input, gradOutput)
	local dimension = self:_getPositiveDimension(input)

	for i=1,#input do
		if self.gradInput[i] == nil then
			self.gradInput[i] = input[i].new()
		end
		self.gradInput[i]:resizeAs(input[i])
	end

	-- clear out invalid gradInputs
	for i=#input+1, #self.gradInput do
		self.gradInput[i] = nil
	end

	local offset = 1
	for i=1,#input do
		local currentOutput = input[i]
		local currentGradInput = gradOutput:narrow(dimension, offset,
							 currentOutput:size(dimension))
		self.gradInput[i]:copy(currentGradInput)
		offset = offset + currentOutput:size(dimension)
	end
	self.output = table.remove(self._output, self.rindex)
	return self.gradInput
end

function aSeqJoinTable:type(type, tensorCache)
	self.gradInput = {}
	return parent.type(self, type, tensorCache)
end
