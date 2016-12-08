--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement an NMT for RNN:

	Version 0.0.4

]]

local aNMT, parent = torch.class('nn.aNMT', 'nn.aBstractNMT')

-- generate a module
function aNMT:__init(encoder, decoder, attention, lookup, classifier, eosid, max_Target_Length, evaOTag)

	parent.__init(self)

	self.encoder = encoder-- a Sequence module input the whole sequence
	self.decoder = decoder-- a Step module process the sequence step by step
	self.attention = attention
	self.lookup = lookup
	self.classifier = classifier
	self.eosid = eosid
	self.maxLength = max_Target_Length or 256
	self.evaOTag = evaOTag

	self.modules = {self.encoder, self.decoder, self.attention}

end

function aNMT:_seq_updateOutput(input)

	-- get source and target
	local src = input[1]
	local tar = input[2]

	-- encode the whole sequence
	self._encoded = self:_tranTable(self.encoder:updateOutput(input))

	self:_copy_forward(encoder, decoder)

	-- prepare the whole sequence
	local _cOutput = self._encoded[-1]
	local _cAttention = self.attention:updateOutput({self._encoded, _cOutput})
	self._initInput = torch.cat(_cOutput, _cAttention)
	local _nInput = self._initInput

	local _output = {}-- store the result

	if self.train then

		for _ = 1, #tar do
			_cOutput = self.decoder:updateOutput(_nInput)
			_cAttention = self.attention:updateOutput({self._encoded, _cOutput})
			_nInput = torch.cat(_cOutput, _cAttention)
			table.insert(_output, _nInput)
		end

	else

		local _length = 1
		while not self.state:all() and _length < self.maxLength do
			_cOutput = self.decoder:updateOutput(_nInput)
			_cAttention = self.attention:updateOutput({self._encoded, _cOutput})
			_nInput = torch.cat(_cOutput, _cAttention)
			local _rs = self.classifier:updateOutput(_nInput)
			self:_updateState(_rs)
			_length = _length + 1
			if self.evaOTag then
				table.insert(_output, _rs)
			else
				table.insert(_output, _nInput)
			end
		end
	end

	self.output = _output

	return self.output

end

function aNMT:_seq_backward(input, gradOutput, scale)

	local tGradInput

	local src = input[1]
	local tar = input[2]

	-- backward the decoder
	local _gEncoder, _gDecoder, __gEncoder
	local _gInput
	for _ = #tar, 1, -1 do

		local _cAttention = table.remove(self.oattention)
		local _cDecoder = table.remove(self.odecoder)
		local _cGrad = table.remove(gradOutput)
		local _cInput = table.remove(tar)

		if _gInput then
			_cGrad:add(_gInput)
		end

		if _gEncoder then
			__gEncoder, _gDecoder = self.attention:backward({self._encoded, _cDecoder}, )
			_gEncoder:add(__gEncoder)
		else
			_gEncoder, _gDecoder = self.attention:backward({self._encoded, _cDecoder}, )
		end

		_gInput = self.decoder:backward(_cInput, _gDecoder, scale)

		if _ > 1 then
			tGradInput[_] = _gInput:clone()
		end

	end

	tGradInput[1] = _gInput

	self:_copy_backward(decoder, encoder)

	local sGradInput = self.encoder:backward(src, self:_tranTensor(_gEncoder), scale)

	return {sGradInput, tGradInput}

end
