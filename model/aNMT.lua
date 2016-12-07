--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement an NMT for RNN:

	Version 0.0.3

]]

local aNMT, parent = torch.class('nn.aNMT', 'nn.aBstractNMT')

-- generate a module
function aNMT:__init(encoder, decoder, attention, lookup, classifier, eosid, target_Length)

	parent.__init(self)

	self.encoder = encoder-- a Sequence module input the whole sequence
	self.decoder = decoder-- a Step module process the sequence step by step
	self.attention = attention
	self.lookup = lookup
	self.classifier = classifier
	self.eosid = eosid
	self.glength = target_Length or 256

	self.modules = {self.encoder, self.decoder, self.attention, self.classifier}

end

function aNMT:_seq_updateOutput(input)

	local _output = {}-- store the result

	if self.train then

		-- get source and target
		local src = input[1]
		local tar = input[2]

		-- encode the whole sequence
		self._encoded = self:_tranTable(self.encoder:updateOutput(src))

		self:_copy_forward(encoder, decoder)

		-- prepare the whole sequence
		local _cPrevOutput = self._encoded[-1]-- still problem here
		for _, v in tar do
			local odecoder = self.decoder:updateOutput(_cPrevOutput)
			local oattention = self.attention:updateOutput({self._encoded, odecoder})
			_cPrevOutput = torch.cat(odecoder, oattention)
			local oclassify = self.classifier:updateOutput(_cPrevOutput)
			table.insert(_output, oclassify)
			table.insert(self.oattention, oattention)
			table.insert(self.odecoder, odecoder)
		end

	else

		-- encode the whole sequence
		self._encoded = self:_tranTable(self.encoder:updateOutput(input))

		self:_copy_forward(encoder, decoder)

		local oclassify = self:_prepare_data(input[1])
		local _length = 1
		while not self.state:all() do
			oclassify = self.classifier:updateOutput(self.attention:updateOutput(self.decoder:updateOutput(self.lookup:updateOutput(oclassify))))
			table.insert(_output, oclassify)
			self:_updateState(oclassify)
			if _length > self.glength then
				break
			end
			_length = _length + 1
		end
	end

	self.output = output

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
