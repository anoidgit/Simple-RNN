--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard vanilla LSTM:

	i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + W[c->i]c[t−1] + b[1->i])      (1)
	f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + W[c->f]c[t−1] + b[1->f])      (2)
	z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
	c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
	o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + W[c->o]c[t] + b[1->o])        (5)
	h[t] = o[t]tanh(c[t])                                                (6)

	Version 0.0.2

]]

local aLSTM, parent = torch.class('nn.aLSTM', 'nn.Container')

function aLSTM:__init(inputSize, outputSize, batchmode, maskZero)

	parent.__init(self)

	if batchmode then
		self.narrowDim = 2
	else
		self.narrowDim = 1
	end

	-- set wether mask zero
	self.maskzero = maskZero

	-- prepare should be only debug use consider of efficient
	self:prepare()

	-- asign the default method
	self:_asign()

	-- forget gate start index
	-- also was used in updateOutput to prepare init cell and output,
	-- because it is outputSize + 1, take care of this
	self.fgstartid = outputSize + 1

	self.inputSize, self.outputSize = inputSize, outputSize

	self:reset()

end

-- asign default method
function aLSTM:_asign()

	self.updateOutput = self._seq_updateOutput
	self.backward = self._seq_backward
	self.updateGradInput = self._seq_updateGradInput

end

-- updateOutput called by forward,
-- It input a time step's input and produce an output
function aLSTM:_step_updateOutput(input)

	-- ensure cell and output are ready for the first step
	if not self.cell then
		-- set batch size and prepare the cell and output
		local _nIdim = input:nDimension()
		if _nIdim>1 then
			self.batchsize = input:size(1)
			self.cell0 = torch.repeatTensor(self.bias:narrow(1, 1, self.outputSize), self.batchsize, 1)
			self.output0 = torch.repeatTensor(self.bias:narrow(1, self.fgstartid, self.outputSize), self.batchsize, 1)
			-- narrow dimension
			self.narrowDim = _nIdim
		else
			self.batchsize = nil
			self.cell0 = self.bias:narrow(1, 1, self.outputSize)
			self.output0 = self.bias:narrow(1, self.fgstartid, self.outputSize)
			-- narrow dimension
			self.narrowDim = 1
		end
		self.cell = self.cell0
		self.output = self.output0
	end

	-- compute input gate and forget gate
	local _ifgo = self.ifgate:forward({input, self.output, self.cell})

	-- get input gate and forget gate
	local _igo = _ifgo:narrow(self.narrowDim, 1, self.outputSize)
	local _fgo = _ifgo:narrow(self.narrowDim, self.fgstartid, self.outputSize)

	-- compute update
	local _zo = self.zmod:forward({input, self.output})

	-- get new value of the cell
	self.cell = torch.add(torch.cmul(_fgo, self.cell),torch.cmul(_igo,_zo))

	-- compute output gate with the new cell,
	-- this is the standard lstm,
	-- otherwise it can be computed with input gate and forget gate
	local _ogo = self.ogate:forward({input, self.output, self.cell})

	-- compute the final output for this input
	local _otanh = torch.tanh(self.cell)
	self.output = torch.cmul(_ogo, _otanh)

	-- if training, remember what should remember
	if self.train then
		table.insert(self._cell, self.cell)
		table.insert(self._output, self.output)
		table.insert(self.otanh, _otanh)
		table.insert(self.ofgate, _fgo)
		table.insert(self.ougate, _zo)
	end

	-- return the output for this input
	return self.output

end

-- updateOutput for a table sequence
function aLSTM:_seq_updateOutput(input)

	local output = {}

	-- ensure cell and output are ready for the first step
	-- set batch size and prepare the cell and output
	local _nIdim = input[1]:nDimension()
	if _nIdim>1 then
		self.batchsize = input[1]:size(1)
		self.cell0 = torch.repeatTensor(self.bias:narrow(1, 1, self.outputSize), self.batchsize, 1)
		self.output0 = torch.repeatTensor(self.bias:narrow(1, self.fgstartid, self.outputSize), self.batchsize, 1)
		-- narrow dimension
		self.narrowDim = _nIdim
	else
		self.batchsize = nil
		self.cell0 = self.bias:narrow(1, 1, self.outputSize)
		self.output0 = self.bias:narrow(1, self.fgstartid, self.outputSize)
		-- narrow dimension
		self.narrowDim = 1
	end
	self.cell = self.cell0
	local _output = self.output0

	-- forward the whole sequence
	for _,iv in ipairs(input) do
		-- compute input gate and forget gate
		local _ifgo = self.ifgate:forward({iv, _output, self.cell})

		-- get input gate and forget gate
		local _igo = _ifgo:narrow(self.narrowDim, 1, self.outputSize)
		local _fgo = _ifgo:narrow(self.narrowDim, self.fgstartid, self.outputSize)

		-- compute update
		local _zo = self.zmod:forward({iv, _output})

		-- get new value of the cell
		self.cell = torch.add(torch.cmul(_fgo, self.cell),torch.cmul(_igo,_zo))

		-- compute output gate with the new cell,
		-- this is the standard lstm,
		-- otherwise it can be computed with input gate and forget gate
		local _ogo = self.ogate:forward({iv, _output, self.cell})

		-- compute the final output for this input
		local _otanh = torch.tanh(self.cell)
		_output = torch.cmul(_ogo, _otanh)

		table.insert(output, _output)

		-- if training, remember what should remember
		if self.train then
			table.insert(self._cell, self.cell)
			table.insert(self._output, _output)
			table.insert(self.otanh, _otanh)
			table.insert(self.ofgate, _fgo)
			table.insert(self.ougate, _zo)
		end

	end

	--[[for _,v in ipairs(input) do
		table.insert(output,self:_step_updateOutput(v))
	end]]

	-- this have conflict with _step_updateOutput,
	-- but anyhow do not use them at the same time
	self.output = output

	return self.output

end

-- backward for one step,
-- though I do not know when to use this
function aLSTM:_step_backward(input, gradOutput, scale)

	-- if need to mask zero, then mask
	if self.maskzero then
		self:_step_makeZero(input, gradOutput)
	end

	local gradInput

	-- grad to cell, gate and input
	local _gCell, _gg, gradInput

	-- if this is not the last step
	if self.__gLOutput then

		-- if this is not the first step
		if #self._output > 0 then

			-- add gradOutput of the sequence behind
			gradOutput:add(self._gLOutput)

			local _cInput = table.remove(input)-- current input
			local _cPrevOutput = table.remove(self._output)-- previous output
			local _cPrevCell = table.remove(self._cell)-- previous cell

			local _cotanh = table.remove(self.otanh)-- output of the tanh after cell for the final output
			local _cofgate = table.remove(self.ofgate)-- output of the forget gate
			local _cougate = table.remove(self.ougate)-- output of the update gate

			-- backward

			-- grad to output gate
			_gg = torch.cmul(gradOutput, _cotanh)

			-- backward output gate
			gradInput, self._gLOutput, _gCell = unpack(self.ogate:backward({_cInput, _cPrevOutput, self.cell}, _gg, scale))

			-- add gradOutput of the sequence behind
			_gCell:add(self._gLCell)

			-- backward from the output tanh to cell
			_gCell:add(self:_tanh_updateGradInput(self.cell, torch.cmul(gradOutput, _cPrevOutput)))

			-- backward update gate
			local __gInput, __gLOutput = unpack(self.zmod:backward({_cInput, _cPrevOutput}, torch.cmul(_gCell, _cougate), scale))
			gradInput:add(__gInput)
			self._gLOutput:add(__gLOutput)

			-- compute the gradOutput of the Prev cell
			self._gLCell = torch.cmul(_gCell, _cofgate)

			-- backward ifgate(input and forget gate)
			-- compute gradOutput
			_gg:resize(self.batchsize, 2 * self.outputSize)
			_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_gCell, _cPrevCell))
			_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gCell, _cougate))
			-- backward the gate
			__gInput, __gLOutput, __gLCell = unpack(self.ifgate:backward({_cInput, _cPrevOutput, _cPrevCell}, _gg, scale))
			gradInput:add(__gInput)
			self._gLOutput:add(__gLOutput)
			self._gLCell:add(__gLCell)

			-- move self.cell(current cell) ahead
			self.cell = _cPrevCell

		else

			-- for the first step

			-- add gradOutput of the sequence behind
			gradOutput:add(self._gLOutput)

			local _cInput = table.remove(input)-- current input
			local _cPrevOutput = self.output0-- previous output
			local _cPrevCell = self.cell0-- previous cell

			local _cotanh = table.remove(self.otanh)-- output of the tanh after cell for the final output
			local _cofgate = table.remove(self.ofgate)-- output of the forget gate
			local _cougate = table.remove(self.ougate)-- output of the update gate

			-- backward

			-- grad to output gate
			_gg = torch.cmul(gradOutput, _cotanh)
			-- backward output gate
			gradInput, self._gLOutput, _gCell = unpack(self.ogate:backward({_cInput, _cPrevOutput, self.cell}, _gg, scale))

			-- add gradOutput of the sequence behind
			_gCell:add(self._gLCell)

			-- backward from the output tanh to cell
			_gCell:add(self:_tanh_updateGradInput(self.cell, torch.cmul(gradOutput, _cPrevOutput)))

			-- backward update gate
			local __gInput, __gLOutput = unpack(self.zmod:backward({_cInput, _cPrevOutput}, torch.cmul(_gCell, _cougate), scale))
			gradInput:add(__gInput)
			self._gLOutput:add(__gLOutput)

			-- compute the gradOutput of the Prev cell
			self._gLCell = torch.cmul(_gCell, _cofgate)

			-- backward ifgate(input and forget gate)
			-- compute gradOutput
			_gg:resize(self.batchsize, 2 * self.outputSize)
			_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_gCell, _cPrevCell))
			_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gCell, _cougate))
			-- backward the gate
			__gInput, __gLOutput, __gLCell = unpack(self.ifgate:backward({_cInput, _cPrevOutput, _cPrevCell}, _gg, scale))
			gradInput:add(__gInput)
			self._gLOutput:add(__gLOutput)
			self._gLCell:add(__gLCell)

			-- accGradParameters for self
			self:_accGradParameters(scale)

			-- prepare for next forward sequence, clear cache
			self:clear()

		end

	else

		-- for the last step

		-- remove the last output
		table.remove(self._output)
		-- get current cell,
		-- it will be used will backward output gate
		self.cell = table.remove(self._cell)

		--backward the last

		-- prepare data for future use
		local _cInput = table.remove(input)-- current input
		local _cPrevOutput = table.remove(self._output)-- previous output
		local _cPrevCell = table.remove(self._cell)-- previous cell

		local _cotanh = table.remove(self.otanh)-- output of the tanh after cell for the final output
		local _cofgate = table.remove(self.ofgate)-- output of the forget gate
		local _cougate = table.remove(self.ougate)-- output of the update gate

		-- backward

		-- grad to output gate
		_gg = torch.cmul(gradOutput, _cotanh)

		-- backward output gate
		gradInput, self._gLOutput, _gCell = unpack(self.ogate:backward({_cInput, _cPrevOutput, self.cell}, _gg, scale))

		-- backward from the output tanh to cell
		_gCell:add(self:_tanh_updateGradInput(self.cell, torch.cmul(gradOutput, _cPrevOutput)))

		-- backward update gate
		local __gInput, __gLOutput = unpack(self.zmod:backward({_cInput, _cPrevOutput}, torch.cmul(_gCell, _cougate), scale))
		gradInput:add(__gInput)
		self._gLOutput:add(__gLOutput)

		-- compute the gradOutput of the Prev cell
		self._gLCell = torch.cmul(_gCell, _cofgate)

		-- backward ifgate(input and forget gate)
		-- compute gradOutput
		_gg:resize(self.batchsize, 2 * self.outputSize)
		_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_gCell, _cPrevCell))
		_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gCell, _cougate))
		-- backward the gate
		local __gLCell
		__gInput, __gLOutput, __gLCell = unpack(self.ifgate:backward({_cInput, _cPrevOutput, _cPrevCell}, _gg, scale))
		gradInput:add(__gInput)
		self._gLOutput:add(__gLOutput)
		self._gLCell:add(__gLCell)

		-- move self.cell(current cell) ahead
		self.cell = _cPrevCell

	end

	-- this have conflict with _table_seq_backward,
	-- but anyhow do not use them at the same time
	self.gradInput = gradInput

	return self.gradInput

end

-- backward process the whole sequence
-- it takes the whole input, gradOutput sequence as input
-- and it will clear the cache after done backward
function aLSTM:_seq_backward(input, gradOutput, scale)

	-- if need to mask zero, then mask
	if self.maskzero then
		self:_seq_makeZero(input, gradOutput)
	end

	local _length = #input
	
	local _input = self:_cloneTable(input)

	local gradInput = {}

	-- remove the last output, because it never used
	table.remove(self._output)
	-- get current cell,
	-- it will be used will backward output gate
	self.cell = table.remove(self._cell)

	-- grad to input and cell
	local _gInput, _gCell

	--backward the last

	-- prepare data for future use
	local _cGradOut = table.remove(gradOutput)-- current gradOutput
	local _cInput = table.remove(_input)-- current input
	local _cPrevOutput = table.remove(self._output)-- previous output
	local _cPrevCell = table.remove(self._cell)-- previous cell

	local _cotanh = table.remove(self.otanh)-- output of the tanh after cell for the final output
	local _cofgate = table.remove(self.ofgate)-- output of the forget gate
	local _cougate = table.remove(self.ougate)-- output of the update gate

	-- backward

	-- grad to output gate
	local _gg = torch.cmul(_cGradOut, _cotanh)

	-- backward output gate
	_gInput, self._gLOutput, _gCell = unpack(self.ogate:backward({_cInput, _cPrevOutput, self.cell}, _gg, scale))

	-- backward from the output tanh to cell
	_gCell:add(self:_tanh_updateGradInput(self.cell, torch.cmul(_cGradOut, _cPrevOutput)))

	-- backward update gate
	local __gInput, __gLOutput = unpack(self.zmod:backward({_cInput, _cPrevOutput}, torch.cmul(_gCell, _cougate), scale))
	_gInput:add(__gInput)
	self._gLOutput:add(__gLOutput)

	-- compute the gradOutput of the Prev cell
	self._gLCell = torch.cmul(_gCell, _cofgate)

	-- backward ifgate(input and forget gate)
	-- compute gradOutput
	_gg:resize(self.batchsize, 2 * self.outputSize)
	_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_gCell, _cPrevCell))
	_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gCell, _cougate))
	-- backward the gate
	local __gLCell
	__gInput, __gLOutput, __gLCell = unpack(self.ifgate:backward({_cInput, _cPrevOutput, _cPrevCell}, _gg, scale))
	_gInput:add(__gInput)
	self._gLOutput:add(__gLOutput)
	self._gLCell:add(__gLCell)

	-- move self.cell(current cell) ahead
	self.cell = _cPrevCell

	gradInput[_length] = _gInput:clone()

	-- backward from end to 2
	for _t = _length - 1, 2, -1 do

		-- prepare data for future use
		_cGradOut = table.remove(gradOutput)-- current gradOutput

		-- add gradOutput of the sequence behind
		_cGradOut:add(self._gLOutput)

		_cInput = table.remove(_input)-- current input
		_cPrevOutput = table.remove(self._output)-- previous output
		_cPrevCell = table.remove(self._cell)-- previous cell

		_cotanh = table.remove(self.otanh)-- output of the tanh after cell for the final output
		_cofgate = table.remove(self.ofgate)-- output of the forget gate
		_cougate = table.remove(self.ougate)-- output of the update gate

		-- backward

		-- grad to output gate
		_gg = torch.cmul(_cGradOut, _cotanh)

		-- backward output gate
		_gInput, self._gLOutput, _gCell = unpack(self.ogate:backward({_cInput, _cPrevOutput, self.cell}, _gg, scale))

		-- add gradOutput of the sequence behind
		_gCell:add(self._gLCell)

		-- backward from the output tanh to cell
		_gCell:add(self:_tanh_updateGradInput(self.cell, torch.cmul(_cGradOut, _cPrevOutput)))

		-- backward update gate
		__gInput, __gLOutput = unpack(self.zmod:backward({_cInput, _cPrevOutput}, torch.cmul(_gCell, _cougate), scale))
		_gInput:add(__gInput)
		self._gLOutput:add(__gLOutput)

		-- compute the gradOutput of the Prev cell
		self._gLCell = torch.cmul(_gCell, _cofgate)

		-- backward ifgate(input and forget gate)
		-- compute gradOutput
		_gg:resize(self.batchsize, 2 * self.outputSize)
		_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_gCell, _cPrevCell))
		_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gCell, _cougate))
		-- backward the gate
		__gInput, __gLOutput, __gLCell = unpack(self.ifgate:backward({_cInput, _cPrevOutput, _cPrevCell}, _gg, scale))
		_gInput:add(__gInput)
		self._gLOutput:add(__gLOutput)
		self._gLCell:add(__gLCell)

		-- move self.cell(current cell) ahead
		self.cell = _cPrevCell

		gradInput[_t] = _gInput:clone()
	end

	-- backward for the first time step

	-- prepare data for future use
	_cGradOut = table.remove(gradOutput)-- current gradOutput

	-- add gradOutput of the sequence behind
	_cGradOut:add(self._gLOutput)

	_cInput = table.remove(_input)-- current input
	_cPrevOutput = self.output0-- previous output
	_cPrevCell = self.cell0-- previous cell

	_cotanh = table.remove(self.otanh)-- output of the tanh after cell for the final output
	_cofgate = table.remove(self.ofgate)-- output of the forget gate
	_cougate = table.remove(self.ougate)-- output of the update gate

	-- backward

	-- grad to output gate
	_gg = torch.cmul(_cGradOut, _cotanh)
	-- backward output gate
	_gInput, self._gLOutput, _gCell = unpack(self.ogate:backward({_cInput, _cPrevOutput, self.cell}, _gg, scale))

	-- add gradOutput of the sequence behind
	_gCell:add(self._gLCell)

	-- backward from the output tanh to cell
	_gCell:add(self:_tanh_updateGradInput(self.cell, torch.cmul(_cGradOut, _cPrevOutput)))

	-- backward update gate
	__gInput, __gLOutput = unpack(self.zmod:backward({_cInput, _cPrevOutput}, torch.cmul(_gCell, _cougate), scale))
	_gInput:add(__gInput)
	self._gLOutput:add(__gLOutput)

	-- compute the gradOutput of the Prev cell
	self._gLCell = torch.cmul(_gCell, _cofgate)

	-- backward ifgate(input and forget gate)
	-- compute gradOutput
	_gg:resize(self.batchsize, 2 * self.outputSize)
	_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_gCell, _cPrevCell))
	_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gCell, _cougate))
	-- backward the gate
	__gInput, __gLOutput, __gLCell = unpack(self.ifgate:backward({_cInput, _cPrevOutput, _cPrevCell}, _gg, scale))
	_gInput:add(__gInput)
	self._gLOutput:add(__gLOutput)
	self._gLCell:add(__gLCell)

	gradInput[1] = _gInput

	-- accGradParameters for self
	self:_accGradParameters(scale)

	-- prepare for next forward sequence, clear cache
	self:clear()

	self.gradInput = gradInput

	return self.gradInput

end

function aLSTM:_seq_updateGradInput(input, gradOutput)

	return self:backward(input, gradOutput)

end

-- modules in aLSTM.modules were done while backward
function aLSTM:accGradParameters(input, gradOutput, scale)

	self:_accGradParameters(scale)

end

-- updateParameters 
function aLSTM:updateParameters(learningRate)

	for _, module in ipairs(self.modules) do
		module:updateParameters(learningRate)
	end
	self.bias:add(-learningRate, self.gradBias)

end

-- zeroGradParameters
function aLSTM:zeroGradParameters()

	for _, module in ipairs(self.modules) do
		module:zeroGradParameters()
	end
	self.gradBias:zero()

end

-- accGradParameters used for aLSTM.bias
function aLSTM:_accGradParameters(scale)

	scale = scale or 1
	if self.batchsize then
		self._gLCell = self._gLCell:sum(self.narrowDim)
		self._gLCell:resize(self.outputSize)
		self._gLOutput = self._gLOutput:sum(self.narrowDim)
		self._gLOutput:resize(self.outputSize)
	end
	self.gradBias:narrow(1,1,self.outputSize):add(scale, self._gLCell)
	self.gradBias:narrow(1,self.fgstartid,self.outputSize):add(scale, self._gLOutput)

end

-- clear the storage
function aLSTM:clear()

	-- cell sequence
	self._cell = {}
	-- last cell
	self.cell = nil
	-- output sequence
	self._output = {}
	-- last output
	self.output = nil
	-- gradInput sequence
	self.gradInput = nil

	-- after tanh value for the final output from cell
	self.otanh = {}
	-- output of the forget gate
	self.ofgate = {}
	-- output of the update gate
	self.ougate = {}

	-- grad from the sequence after
	self._gLCell = nil
	self._gLOutput = nil

end

-- define type
function aLSTM:type(type, ...)
	return parent.type(self, type, ...)
end

-- evaluate
function aLSTM:evaluate()
	self.train = false
	self:clear()
	for _, module in ipairs(self.modules) do
		module:evaluate()
	end
end

-- train
function aLSTM:training()
	self.train = true
	for _, module in ipairs(self.modules) do
		module:training()
	end
end

-- reset the module
function aLSTM:reset()

	self.ifgate = self:buildIFModule()
	self.zmod = self:buildUpdateModule()
	self.ogate = self:buildOGModule()

	--[[ put the modules in self.modules,
	so the default method could be done correctly]]
	self.modules = {self.ifgate, self.zmod, self.ogate}

	-- inner parameters need to correctly processed
	-- in fact, it is output and cell at time step 0
	self.bias = torch.zeros(2 * self.outputSize)
	self.gradBias = self.bias:clone()

	self:clear()

end

-- build input and forget gate
function aLSTM:buildIFModule()

	local _ifm = nn.aSequential()
		:add(nn.aJoinTable(self.narrowDim,self.narrowDim))
		:add(nn.aLinear(self.inputSize + self.outputSize * 2, self.outputSize * 2))
		:add(nn.aSigmoid())

	return _ifm

end

-- build output gate
function aLSTM:buildOGModule()

	local _ogm = nn.aSequential()
		:add(nn.aJoinTable(self.narrowDim,self.narrowDim))
		:add(nn.aLinear(self.inputSize + self.outputSize * 2, self.outputSize))
		:add(nn.aSigmoid())

	return _ogm

end

-- build update module
function aLSTM:buildUpdateModule()

	local _um = nn.aSequential()
		:add(nn.aJoinTable(self.narrowDim,self.narrowDim))
		:add(nn.aLinear(self.inputSize + self.outputSize, self.outputSize))
		:add(nn.aTanh())

	return _um

end

-- prepare for LSTM
function aLSTM:prepare()

	-- Warning: This method may be DEPRECATED at any time
	-- it is for debug use
	-- you should write a faster and simpler module instead of nn
	-- for your particular use

	nn.aJoinTable = nn.JoinTable
	nn.aLinear = nn.Linear
	--nn.aTanh = nn.Tanh
	require "aTanh"
	nn.aSigmoid = nn.Sigmoid
	nn.aSequential = nn.Sequential

end

-- mask zero for a step
function aLSTM:_step_makeZero(input, gradOutput)

	if self.batchsize then
		-- if batch input
		
		-- get a zero unit
		local _stdZero = input:narrow(1,1,1):clone():zero()
		-- look at each unit
		for _t = 1, self.batchsize do
			if input:narrow(1,_t,1):equal(_stdZero) then
				-- if it was zero, then zero the gradOutput
				gradOutput:narrow(1,_t,1):zero()
			end
		end
	else
		-- if not batch

		local _stdZero = input:clone():zero()
		if input:equal(_stdZero) then
			gradOutput:zero()
		end
	end

end

-- mask zero for a sequence
function aLSTM:_seq_makeZero(input, gradOutput)

	-- walk the whole sequence
	for _t,v in ipairs(input) do
		-- make zero for each step
		self:_step_makeZero(v,gradOutput[_t])
	end

end

-- copy a table
function aLSTM:_cloneTable(tbsrc)

	local tbrs = {}
	for k,v in ipairs(tbsrc) do
		tbrs[k] = v
	end

	return tbrs
end

-- calc gradInput of tanh
function aLSTM:_tanh_updateGradInput(input, gradOutput)

	local _gradInput = input.new()
	_gradInput:resizeAs(input):fill(1)
	_gradInput:addcmul(-1,gradOutput,gradOutput)

	return _gradInput

end

-- introduce self
function aLSTM:__tostring__()

	return string.format('%s(%d -> %d)', torch.type(self), self.inputSize, self.outputSize)

end
