--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement an NMT based RNN:

	Version 0.0.15

]]

local aGlobalAttentionNMT, parent = torch.class('nn.aGlobalAttentionNMT', 'nn.aBstractNMT')

-- generate a module
function aGlobalAttentionNMT:__init(encoder, decoder, attention, classifier, eosid, max_Target_Length, maskzero, evaOTag)

	parent.__init(self)

	self.encoder = encoder-- a Sequence module input the whole sequence
	self.decoder = decoder-- a Step module process the sequence step by step
	self.attention = attention
	self.classifier = classifier
	self.eosid = eosid
	self.maxLength = max_Target_Length or 256
	self.maskzero = maskzero
	self.evaOTag = evaOTag

	self.modules = {self.encoder, self.decoder, self.attention}

end

function aGlobalAttentionNMT:clearState()

	self._initInput = nil
	self._encoded = nil

end

function aGlobalAttentionNMT:forget()

	self:clearState()
	for _, module in ipairs(self.modules) do
		module:forget()
	end

end

function aGlobalAttentionNMT:_seq_updateOutput(input)

	self.gradInput = nil

	if not self.train then
		self:forget()
	end

	-- get source and target
	local src = input[1]
	local tar = input[2]

	-- encode the whole sequence
	--self._encoded = self:_tranTable(self.encoder:updateOutput(src))
	self._encoded = self.encoder:updateOutput(src)

	--self:_copy_forward(self.encoder, self.decoder)

	-- prepare the whole sequence
	local _cOutput = self._encoded[-1]
	local _cAttention = self.attention:updateOutput({self._encoded, _cOutput})
	self._initInput = {_cOutput, _cAttention}
	local _nInput = self._initInput

	local _output = {}-- store the result

	if self.train or #tar > 0 then

		for _, v in ipairs(tar) do
			_cOutput = self.decoder:updateOutput(_nInput)
			if self.maskzero then
				self:_maskDecoderZero(v, _cOutput)
			end
			_cAttention = self.attention:updateOutput({self._encoded, _cOutput})
			_nInput = {_cOutput, _cAttention}
			table.insert(_output, _nInput)
		end

	else

		local _length = 1
		while not self.state:all() and _length < self.maxLength do
			_cOutput = self.decoder:updateOutput(_nInput)
			_cAttention = self.attention:updateOutput({self._encoded, _cOutput})
			_nInput = {_cOutput, _cAttention}
			local _rs = self:_maxGetClass(self.classifier:updateOutput(_nInput))
			self:_updateState(_rs)
			if self.evaOTag then
				table.insert(_output, _rs)
			else
				table.insert(_output, _nInput)
			end
			_length = _length + 1
		end

	end

	self.output = _output
	return self.output

end

function aGlobalAttentionNMT:_seq_backward(input, gradOutput, scale)

	local src = input[1]
	local tar = input[2]

	local _gEncoder

	local _cOutput = table.remove(self.output)[1]

	local _gLOutput, _gLAttention

	local __gEncoder, _gOutput

	for _ = #tar, 1, -1 do

		local _prevOutput

		if _ > 1 then
			_prevOutput, _prevAttention = unpack(table.remove(self.output))
		else
			_prevOutput, _prevAttention = unpack(self._initInput)
		end

		local _cGradOutput, _cGradAttention = unpack(table.remove(gradOutput))

		if self.maskzero then
			self:_maskDecoderZero(tar[_], _cGradOutput)
		end

		if _gLOutput then
			_cGradOutput:add(_gLOutput)
			_cGradAttention:add(_gLAttention)
		end

		__gEncoder, _gOutput = unpack(self.attention:backward({self._encoded, _cOutput}, _cGradAttention, scale))

		if _gEncoder then
			_gEncoder:add(__gEncoder)
		else
			_gEncoder = __gEncoder
		end

		_cGradOutput:add(_gOutput)

		_gLOutput, _gLAttention = unpack(self.decoder:backward({_prevOutput, _prevAttention}, _cGradOutput, scale))

		_cOutput = _prevOutput

	end

	self.output = nil

	__gEncoder, _gOutput = unpack(self.attention:backward({self._encoded, self._encoded[-1]}, _gLAttention, scale))

	if _gEncoder then
		_gEncoder:add(__gEncoder)
	else
		_gEncoder = __gEncoder
	end

	_gEncoder[-1]:add(_gLOutput):add(_gOutput)

	--self:_copy_backward(decoder, encoder)

	--self.gradInput = {self.encoder:backward(src, self:_tranTensor(_gEncoder), scale), self:_gRecursiveZero(tar)}
	self.gradInput = {self.encoder:backward(src, _gEncoder, scale), self:_gRecursiveZero(tar)}

	self:forget()

	return self.gradInput

end
