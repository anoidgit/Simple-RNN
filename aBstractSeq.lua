--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement an aBstractSeq for RNN:

	Version 0.0.2

]]

local aBstractSeq, parent = torch.class('nn.aBstractSeq', 'nn.Container')

-- generate a module
function aBstractSeq:__init()

	parent.__init(self)

end

-- fit torch standard,
-- calc an output for an input,
-- faster while you just specify which function to use
function aBstractSeq:updateOutput(input)

	if torch.type(input) == 'table' then
		self.tablesequence = true
		self:_table_clearState()
		return self:_seq_updateOutput(input)
	else
		self.tablesequence = nil
		self:_tensor_clearState(input)
		return self:_tseq_updateOutput(input)
	end

end

-- fit torch standard,
-- backward,
-- faster while you just specify which function to use
function aBstractSeq:backward(input, gradOutput, scale)

	if torch.type(input) == 'table' then
		return self:_seq_backward(input, gradOutput, scale)
	else
		return self:_tseq_backward(input, gradOutput, scale)
	end

end

-- fit torch rnn standard,
-- clear the cache used
function aBstractSeq:clearState()

	if self.tablesequence then
		self:_table_clearState()
	else
		self:_tensor_clearState()
	end

	--[[for _, module in ipairs(self.modules) do
		module:clearState()
	end]]

end

-- updateGradInput for sequence,
-- in fact, it call backward
function aBstractSeq:updateGradInput(input, gradOutput)

	return self:backward(input, gradOutput)

end

-- modules in aBstractSeq.modules were done while backward
function aBstractSeq:accGradParameters(input, gradOutput, scale)

	if self.rememberState then

		if self.firstSequence then

			-- accGradParameters for self
			self:_accGradParameters(scale)
			self.firstSequence = false

		end

	else

		self:_accGradParameters(scale)

	end

end

-- evaluate
function aBstractSeq:evaluate()

	self.train = false

	for _, module in ipairs(self.modules) do
		module:evaluate()
	end

	self:forget()

end

-- train
function aBstractSeq:training()

	self.train = true

	for _, module in ipairs(self.modules) do
		module:training()
	end

	self:forget()

end

-- remember last state or not
function aBstractSeq:remember(mode)

	-- set default to both
	local _mode = mode or "both"

	if _mode == "both" or _mode == true then
		self.rememberState = true
	else
		self.rememberState = nil
	end

	self:forget()

end

-- mask zero for a sequence
function aBstractSeq:_seq_makeZero(input, gradOutput)

	-- get a storage
	local _fi = input[1]
	local _stdZero = _fi.new()

	if self.batchsize then
	-- if batch input

		-- get a zero unit
		_stdZero:resizeAs(_fi[1]):zero()

		-- walk the whole sequence
		for _t,v in ipairs(input) do
			-- make zero for each step
			-- look at each unit
			local _grad = gradOutput[_t]
			for __t = 1, self.batchsize do
				if v[__t]:equal(_stdZero) then
					-- if it was zero, then zero the gradOutput
					_grad[__t]:zero()
				end
			end
		end

	else

		_stdZero:resizeAs(_fi):zero()

		-- walk the whole sequence
		for _t,v in ipairs(input) do
			-- make zero for each step
			-- look at each unit
			if v:equal(_stdZero) then
				-- if it was zero, then zero the gradOutput
				gradOutput[_t]:zero()
			end
		end

	end

end

-- mask zero for a tensor sequence
function aBstractSeq:_tseq_makeZero(input, gradOutput)

	local _fi = input[1][1]
	local iSize = input:size()
	local _stdZero = _fi.new()
	_stdZero:resizeAs(_fi):zero()
	for _i = 1, iSize[1] do
		local _ti = input[_i]
		for _j= 1, iSize[2] do
			if _ti[_j]:equal(_stdZero) then
				gradOutput[_i][_j]:zero()
			end
		end
	end

end

-- copy a table
function aBstractSeq:_cloneTable(tbsrc)

	local tbrs = {}

	for k,v in ipairs(tbsrc) do
		tbrs[k] = v
	end

	return tbrs
end