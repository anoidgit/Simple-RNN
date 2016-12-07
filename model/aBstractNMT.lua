--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement an aBstractNMT for RNN:

	Version 0.0.1

]]

local aBstractNMT, parent = torch.class('nn.aBstractNMT', 'nn.aBstractBase')

-- generate a module
function aBstractNMT:__init()

	parent.__init(self)

end

function aBstractNMT:_prepare_data(input)

	local _iSize = input:size()
	self.state = input.new()
	if _iSize > 1 then
		self.state:resize(_iSize[1])
	else
		self.state:resize(1)
	end
	self.state:zero()
	local eosi = self.state:clone():fill(self.eosid)

	return self.lookup:updateOutput(eosi)

end

-- transfer a table contain data like (batchsize*)inputsize to a tensor with size seqlen(*batchsize)*inputsize
function aBstractNMT:_tranTable(inputTable)

	local _iSize = inputTable[1]:size()
	local _oSize = torch.LongStorage(#_iSize + 1)
	_oSize[1] = 1
	for _ = 1, #_iSize do
		_oSize[_ + 1] = _iSize[_]
	end
	local output = inputTable[1].new()
	output:resize(_oSize)
	for _, v in ipairs(inputTable) do
		output[_]:copy(v)
	end

	return output

end

function aBstractNMT:_tranTensor(input)

	local output = {}
	for _ = 1, input:size(1) do
		table.insert(output, input[_])
	end

	return output

end

function aBstractNMT:_updateState(words)

	local _iSize = words:size()
	local comp
	if #_iSize > 1 then
		comp = words:reshape(_iSize[1])
	else
		comp = words
	end

	self.state[comp:eq(self.eosid)] = 1

end
