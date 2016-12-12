--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement an aBstractNMT for RNN:

	Version 0.0.3

]]

local aBstractNMT, parent = torch.class('nn.aBstractNMT', 'nn.aBstractBase')

-- generate a module
function aBstractNMT:__init()

	parent.__init(self)

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

function aBstractBase:_maxGetClass(scoret)

	local _iSize = scoret:size()
	local _nDim = #_iSize
	local _, c = torch.max(scoret, _nDim)
	if _nDim > 1 then
		c = c:reshape(_iSize[1])
	end
	return c

end

function aBstractNMT:_updateState(words)

	self.state[words:eq(self.eosid)] = 1

end
