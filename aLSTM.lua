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

	Version 0.2.2

]]

local aLSTM, parent = torch.class('nn.aLSTM', 'nn.aBstractSeq')
--local aLSTM, parent = torch.class('nn.aLSTM', 'nn.aBstractStep')

-- generate a module
function aLSTM:__init(inputSize, outputSize, maskZero, remember)

	parent.__init(self)

	-- set wether mask zero
	self.maskzero = maskZero

	-- set wether remember the cell and output between sequence
	-- unless you really need, this was not advised
	-- if you use this,
	-- you need to have the same batchsize until you call forget,
	-- for the last time step,
	-- at least it must be little then the previous step
	self.rememberState = remember

	-- prepare should be only debug use consider of efficient
	--self:prepare()

	-- asign the default method
	-- you can not asign the function,
	-- if you do, the function was saved
	-- so use inner call
	--self:_asign(seqData)

	-- forget gate start index
	-- also was used in updateOutput to prepare init cell and output,
	-- because it is outputSize + 1, take care of this
	self.fgstartid = outputSize + 1

	-- prepare to build the modules
	self.inputSize, self.outputSize = inputSize, outputSize

	self.narrowDim = 1

	self:reset()

end

-- asign default method
--[[function aLSTM:_asign(seqd)

	if seqd then
		self.updateOutput = self._tseq_updateOutput
		self.backward = self._tseq_backward
		self._forget = self._tensor_forget
	else
		self.updateOutput = self._seq_updateOutput
		self.backward = self._seq_backward
		self._forget = self._table_forget
	end
	self.updateGradInput = self._seq_updateGradInput

end]]

-- prepare data for the first step
function aLSTM:_prepare_data(input)

-- set batch size and prepare the cell and output
		local _nIdim = input:nDimension()
		if _nIdim>1 then
			self.batchsize = input:size(1)

			if self.rememberState and self.lastCell then
				if self.lastCell:size(1) == self.batchsize then
					self.cell0 = self.lastCell
					self.output0 = self.lastOutput
				else
					self.cell0 = self.lastCell:narrow(1, 1, self.batchsize)
					self.output0 = self.lastOutput:narrow(1, 1, self.batchsize)
				end
			else
				self.cell0 = self.sbm.bias:narrow(1, 1, self.outputSize)
				self.cell0 = self.cell0:reshape(1,self.outputSize):expand(self.batchsize, self.outputSize)
				self.output0 = self.sbm.bias:narrow(1, self.fgstartid, self.outputSize)
				self.output0 = self.output0:reshape(1,self.outputSize):expand(self.batchsize, self.outputSize)
			end

			-- narrow dimension
			self.narrowDim = _nIdim

		else

			self.batchsize = nil

			if self.rememberState and self.lastCell then
				self.cell0 = self.lastCell
				self.output0 = self.lastOutput
			else
				self.cell0 = self.sbm.bias:narrow(1, 1, self.outputSize)
				self.output0 = self.sbm.bias:narrow(1, self.fgstartid, self.outputSize)
			end

			-- narrow dimension
			self.narrowDim = 1
		end

end

-- updateOutput called by forward,
-- It input a time step's input and produce an output
function aLSTM:_step_updateOutput(input)

	-- ensure cell and output are ready for the first step
	if not self.cell then

		self:_prepare_data(input)

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

-- updateOutput for tensor input,
-- input tensor is expected to be seqlen * batchsize * vecsize
function aLSTM:_tseq_updateOutput(input)

	self.gradInput:resize(0)

	-- get input and output size
	local iSize = input:size()
	local oSize = input:size()
	oSize[3] = self.outputSize
	self.output = input.new()
	self.output:resize(oSize)

	if self.train then

		self._cell:resize(oSize)
		self.otanh:resize(oSize)
		self.oogate:resize(oSize)
		self.ohid:resize(oSize)
		local dOSize = input:size()
		dOSize[3] = self.outputSize * 2
		self.oifgate:resize(dOSize)

	end

	self:_prepare_data(input[1])

	self.cell = self.cell0
	local _output = self.output0

	-- forward the whole sequence
	for _t = 1, iSize[1] do
		local iv = input[_t]
		-- compute input gate and forget gate
		local _ifgo = self.ifgate:forward({iv, _output, self.cell})

		-- get input gate and forget gate
		local _igo = _ifgo:narrow(self.narrowDim, 1, self.outputSize)
		local _fgo = _ifgo:narrow(self.narrowDim, self.fgstartid, self.outputSize)

		-- compute hidden
		local _zo = self.zmod:forward({iv, _output})

		-- get new value of the cell
		self.cell = torch.add(torch.cmul(_fgo, self.cell),torch.cmul(_igo,_zo))

		-- compute output gate with the new cell,
		-- this is the standard lstm,
		-- otherwise it can be computed with input gate and forget gate
		local _ogo = self.ogate:forward({iv, _output, self.cell})

		-- compute the final output for this input
		local _otanh = self.tanh:forward(self.cell)
		_output = torch.cmul(_ogo, _otanh)

		self.output[_t]:copy(_output)

		-- if training, remember what should remember
		if self.train then
			self._cell[_t]:copy(self.cell)--c[t]
			self.otanh[_t]:copy(_otanh)--tanh[t]
			self.oifgate[_t]:copy(_ifgo)--if[t], input and forget
			self.ohid[_t]:copy(_zo)--z[t]
			self.oogate[_t]:copy(_ogo)--o[t]
		end

	end

	return self.output

end

-- updateOutput for a table sequence
function aLSTM:_seq_updateOutput(input)

	self.gradInput = nil

	local output = {}

	self:_prepare_data(input[1])

	self.cell = self.cell0
	local _output = self.output0

	-- forward the whole sequence
	for _,iv in ipairs(input) do
		-- compute input gate and forget gate
		local _ifgo = self.ifgate:forward({iv, _output, self.cell})

		-- get input gate and forget gate
		local _igo = _ifgo:narrow(self.narrowDim, 1, self.outputSize)
		local _fgo = _ifgo:narrow(self.narrowDim, self.fgstartid, self.outputSize)

		-- compute hidden
		local _zo = self.zmod:forward({iv, _output})

		-- get new value of the cell
		self.cell = torch.add(torch.cmul(_fgo, self.cell),torch.cmul(_igo,_zo))

		-- compute output gate with the new cell,
		-- this is the standard lstm,
		-- otherwise it can be computed with input gate and forget gate
		local _ogo = self.ogate:forward({iv, _output, self.cell})

		-- compute the final output for this input
		local _otanh = self.tanh:forward(self.cell)
		_output = torch.cmul(_ogo, _otanh)

		table.insert(output, _output)

		-- if training, remember what should remember
		if self.train then
			table.insert(self._cell, self.cell)--c[t]
			--table.insert(self._output, _output)--h[t]
			table.insert(self.otanh, _otanh)--tanh[t]
			table.insert(self.oifgate, _ifgo)--if[t], input and forget
			table.insert(self.ohid, _zo)--z[t]
			table.insert(self.oogate, _ogo)--o[t]
		end

	end

	--[[for _,v in ipairs(input) do
		table.insert(output,self:_step_updateOutput(v))
	end]]

	-- this have conflict with _step_updateOutput,
	-- but anyhow do not use them at the same time
	self.output = output
	self._output = output

	--[[if self.train then
		self:_check_table_same(self._cell)
		self:_check_table_same(self._output)
		self:_check_table_same(self.otanh)
		self:_check_table_same(self.oifgate)
		self:_check_table_same(self.ohid)
		self:_check_table_same(self.oogate)
	end]]

	return self.output

end

-- This function was used to check whether the first and second item of a table was same,
-- It was only used during debug time,
-- to prevent all element of a table point to the same thing
--[[function aLSTM:_check_table_same(tbin)

	local _rs = true

	if #tbin>2 then
		if tbin[1]:equal(tbin[2]) then
			_rs = false
		end
	end

	if _rs then
		print("pass")
	end

	return _rs

end]]

-- backward for one step,
-- though I do not know when to use this
function aLSTM:_step_backward(input, gradOutput, scale)

	-- if need to mask zero, then mask
	if self.maskzero then
		self:_step_makeZero(input, gradOutput)
	end

	local gradInput

	local nfirstep = true

	-- grad to cell, gate and input
	local _gCell, _gg, gradInput

	-- preclaim to clear code
	local _cPrevOutput, _cPrevCell, _cotanh, _coogate, _coz, _coifgate, _coigate, _cofgate

	-- if this is not the last step
	if self._gLOutput then

		-- add gradOutput from the sequence behind
		gradOutput:add(self._gLOutput)

	else

		-- remove the last output
		local _lastOutput = table.remove(self._output)
		-- get current cell,
		-- it will be used will backward output gate
		self.cell = table.remove(self._cell)

		-- if need to remember to use for the next sequence
		if self.rememberState then
			self.lastCell = self.cell
			self.lastOutput = _lastOutput
		end

	end

	if #self._output > 0 then
		_cPrevOutput = table.remove(self._output)-- previous output
		_cPrevCell = table.remove(self._cell)-- previous cell
	else
		_cPrevOutput = self.output0
		_cPrevCell = self.cell0
		nfirstep = false
	end

	_cotanh = table.remove(self.otanh)-- output of the tanh after cell for the final output
	_coogate = table.remove(self.oogate)-- output of the forget gate
	_coz = table.remove(self.ohid)-- hidden unit produced by input, z[t]
	_coifgate = table.remove(self.oifgate)-- output of the input and forget gate, if[t], input and forget

	-- asign output of input gate and output gate
	_coigate = _coifgate:narrow(self.narrowDim, 1, self.outputSize)-- i[t]
	_cofgate = _coifgate:narrow(self.narrowDim, self.fgstartid, self.outputSize)--f[t] 

	-- backward

	-- grad to output gate
	_gg = torch.cmul(gradOutput, _cotanh)

	-- backward output gate
	gradInput, self._gLOutput, _gCell = unpack(self.ogate:backward({input, _cPrevOutput, self.cell}, _gg, scale))

	if self.__gLCell then
		-- add gradOutput from the sequence behind
		_gCell:add(self._gLCell)
	end

	-- backward from the output tanh to cell
	_gCell:add(self.tanh:updateGradInput(self.cell, torch.cmul(gradOutput, _coogate)))

	-- backward update gate
	local __gInput, __gLOutput = unpack(self.zmod:backward({input, _cPrevOutput}, torch.cmul(_gCell, _coigate), scale))
	gradInput:add(__gInput)
	self._gLOutput:add(__gLOutput)

	-- compute the gradOutput of the Prev cell
	self._gLCell = torch.cmul(_gCell, _cofgate)

	-- backward ifgate(input and forget gate)
	-- compute gradOutput
	_gg:resize(self.batchsize, 2 * self.outputSize)
	_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_gCell, _coz))
	_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gCell, _cPrevCell))
	-- backward the gate
	local __gLCell
	__gInput, __gLOutput, __gLCell = unpack(self.ifgate:backward({input, _cPrevOutput, _cPrevCell}, _gg, scale))
	gradInput:add(__gInput)

	if nfirstep then
		self._gLOutput:add(__gLOutput)
		self._gLCell:add(__gLCell)
		-- move self.cell(current cell) ahead
		self.cell = _cPrevCell
	else
		-- accGradParameters for self
		if self.rememberState then

			if self.firstSequence then
				-- accGradParameters for self
				self._gLOutput:add(__gLOutput)
				self._gLCell:add(__gLCell)
				self:_accGradParameters(scale)
				self.firstSequence = false
			end

			-- prepare for next forward sequence, clear cache
			self:clearState()

		else

			self._gLOutput:add(__gLOutput)
			self._gLCell:add(__gLCell)
			self:_accGradParameters(scale)

			self:forget()

		end
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

	self.output = nil

	-- if need to mask zero, then mask
	if self.maskzero then
		self:_seq_makeZero(input, gradOutput)
	end

	local _length = #input

	-- reference clone the input table,
	-- otherwise it will be cleaned during backward
	local _input = self:_cloneTable(input)

	local gradInput = {}

	-- remove the last output, because it was never used
	local _lastOutput = table.remove(self._output)
	-- get current cell,
	-- it will be used will backward output gate
	self.cell = table.remove(self._cell)--c[t]

	-- remember the end of sequence for next input use
	if self.rememberState then
		self.lastCell = self.cell
		self.lastOutput = _lastOutput
	end

	-- grad to input and cell
	local _gInput, _gCell

	-- pre claim the local variable, they were discribed where they were used.
	local _cGradOut, _cInput, _cPrevOutput, _cPrevCell, _cotanh, _coifgate, _coogate, _coz, _coigate, _cofgate, _gg
	local __gLCell

	local __gInput, __gLOutput

	-- backward from end to 1
	for _t = _length, 1, -1 do

		-- prepare data for future use
		_cGradOut = table.remove(gradOutput)-- current gradOutput

		if self._gLOutput then
			-- add gradOutput from the sequence behind
			_cGradOut:add(self._gLOutput)
		end

		_cInput = table.remove(_input)-- current input
		if _t>1 then
			_cPrevOutput = table.remove(self._output)-- previous output
			_cPrevCell = table.remove(self._cell)-- previous cell
		else
			_cPrevOutput = self.output0
			_cPrevCell = self.cell0
		end

		_cotanh = table.remove(self.otanh)-- output of the tanh after cell for the final output
		_coogate = table.remove(self.oogate)-- output of the output gate
		_coz = table.remove(self.ohid)-- hidden unit produced by input
		_coifgate = table.remove(self.oifgate)-- output of the input and forget gate

		-- asign output of input gate and output gate
		_coigate = _coifgate:narrow(self.narrowDim, 1, self.outputSize) 
		_cofgate = _coifgate:narrow(self.narrowDim, self.fgstartid, self.outputSize) 

		-- backward

		-- grad to output gate
		_gg = torch.cmul(_cGradOut, _cotanh)

		-- backward output gate
		_gInput, self._gLOutput, _gCell = unpack(self.ogate:backward({_cInput, _cPrevOutput, self.cell}, _gg, scale))

		-- backward from the output tanh to cell
		_gCell:add(self.tanh:updateGradInput(self.cell, torch.cmul(_cGradOut, _coogate)))

		if self._gLCell then
			-- add gradOutput from the sequence behind
			_gCell:add(self._gLCell)
		end

		-- backward hidden
		__gInput, __gLOutput = unpack(self.zmod:backward({_cInput, _cPrevOutput}, torch.cmul(_gCell, _coigate), scale))
		_gInput:add(__gInput)
		self._gLOutput:add(__gLOutput)

		-- compute the gradOutput of the Prev cell
		self._gLCell = torch.cmul(_gCell, _cofgate)

		-- backward ifgate(input and forget gate)
		-- compute gradOutput
		_gg:resize(self.batchsize, 2 * self.outputSize)
		_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_gCell, _coz))
		_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gCell, _cPrevCell))
		-- backward the gate
		__gInput, __gLOutput, __gLCell = unpack(self.ifgate:backward({_cInput, _cPrevOutput, _cPrevCell}, _gg, scale))
		_gInput:add(__gInput)

		if _t > 1 then

			self._gLOutput:add(__gLOutput)
			self._gLCell:add(__gLCell)

			gradInput[_t] = _gInput:clone()

			-- move self.cell(current cell) ahead
			self.cell = _cPrevCell

		else

			gradInput[_t] = _gInput

		end

	end

	-- accGradParameters for self
	if self.rememberState then

		if self.firstSequence then
			-- accGradParameters for self
			self._gLOutput:add(__gLOutput)
			self._gLCell:add(__gLCell)
			self:_accGradParameters(scale)
			self.firstSequence = false
		end

		-- prepare for next forward sequence, clear cache
		self:clearState()

	else

		self._gLOutput:add(__gLOutput)
		self._gLCell:add(__gLCell)
		self:_accGradParameters(scale)

		self:forget()

	end


	self.gradInput = gradInput

	return self.gradInput

end

-- backward for tensor input and gradOutput sequence
function aLSTM:_tseq_backward(input, gradOutput, scale)

	-- if need to mask zero, then mask
	if self.maskzero then
		self:_tseq_makeZero(input, gradOutput)
	end

	local iSize = input:size()
	local oSize = gradOutput:size()

	local _length = iSize[1]

	local gradInput = input.new()
	gradInput:resize(iSize)

	-- remove the last output, because it was never used
	local _lastOutput = self.output[_length]
	-- get current cell,
	-- it will be used will backward output gate
	self.cell = self._cell[_length]--c[t]

	-- remember the end of sequence for next input use
	if self.rememberState then
		-- clone it, for fear that self.lastCell and self.lastOutput marks the whole memory of self.cell and self.output as used
		self.lastCell = self.cell:clone()
		self.lastOutput = _lastOutput:clone()
	end

	-- grad to input and cell
	local _gInput, _gCell

	-- pre claim the local variable, they were discribed where they were used.
	local _cGradOut, _cInput, _cPrevOutput, _cPrevCell, _cotanh, _coifgate, _coogate, _coz, _coigate, _cofgate, _gg
	local __gLCell

	local __gInput, __gLOutput

		-- backward from end to 1
	for _t = _length, 1, -1 do

		-- prepare data for future use
		_cGradOut = gradOutput[_t]-- current gradOutput

		if self._gLOutput then
			-- add gradOutput from the sequence behind
			_cGradOut:add(self._gLOutput)
		end

		_cInput = input[_t]-- current input

		if _t > 1 then
			_cPrevOutput = self.output[_t - 1]-- previous output
			_cPrevCell = self._cell[_t - 1]-- previous cell
		else
			_cPrevOutput = self.output0
			_cPrevCell = self.cell0
		end

		_cotanh = self.otanh[_t]-- output of the tanh after cell for the final output
		_coogate = self.oogate[_t]-- output of the output gate
		_coz = self.ohid[_t]-- hidden unit produced by input
		_coifgate = self.oifgate[_t]-- output of the input and forget gate

		-- asign output of input gate and output gate
		_coigate = _coifgate:narrow(self.narrowDim, 1, self.outputSize) 
		_cofgate = _coifgate:narrow(self.narrowDim, self.fgstartid, self.outputSize) 

		-- backward

		-- grad to output gate
		_gg = torch.cmul(_cGradOut, _cotanh)

		-- backward output gate
		_gInput, self._gLOutput, _gCell = unpack(self.ogate:backward({_cInput, _cPrevOutput, self.cell}, _gg, scale))

		-- backward from the output tanh to cell
		_gCell:add(self.tanh:updateGradInput(self.cell, torch.cmul(_cGradOut, _coogate)))

		if self._gLCell then
			-- add gradOutput from the sequence behind
			_gCell:add(self._gLCell)
		end

		-- backward hidden
		__gInput, __gLOutput = unpack(self.zmod:backward({_cInput, _cPrevOutput}, torch.cmul(_gCell, _coigate), scale))
		_gInput:add(__gInput)
		self._gLOutput:add(__gLOutput)

		-- compute the gradOutput of the Prev cell
		self._gLCell = torch.cmul(_gCell, _cofgate)

		-- backward ifgate(input and forget gate)
		-- compute gradOutput
		_gg:resize(self.batchsize, 2 * self.outputSize)
		_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_gCell, _coz))
		_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gCell, _cPrevCell))
		-- backward the gate
		__gInput, __gLOutput, __gLCell = unpack(self.ifgate:backward({_cInput, _cPrevOutput, _cPrevCell}, _gg, scale))
		_gInput:add(__gInput)

		if _t > 1 then
			self._gLOutput:add(__gLOutput)
			self._gLCell:add(__gLCell)

			-- move self.cell(current cell) ahead
			self.cell = _cPrevCell
		end

		gradInput[_t]:copy(_gInput)

	end

	-- accGradParameters for self
	if self.rememberState then

		if self.firstSequence then
			-- accGradParameters for self
			self._gLOutput:add(__gLOutput)
			self._gLCell:add(__gLCell)
			self:_accGradParameters(scale)
			self.firstSequence = false
		end

		-- prepare for next forward sequence, clear cache
		self:clearState()

	else

		self._gLOutput:add(__gLOutput)
		self._gLCell:add(__gLCell)
		self:_accGradParameters(scale)

		self:forget()

	end

	self.gradInput:resizeAs(gradInput):copy(gradInput)

	return self.gradInput

end

-- updateParameters 
--[[function aLSTM:updateParameters(learningRate)

	for _, module in ipairs(self.modules) do
		module:updateParameters(learningRate)
	end
	self.sbm.bias:add(-learningRate, self.sbm.gradBias)

end]]

-- zeroGradParameters
--[[function aLSTM:zeroGradParameters()

	for _, module in ipairs(self.modules) do
		module:zeroGradParameters()
	end
	self.sbm.gradBias:zero()

end]]

-- accGradParameters used for aLSTM.bias
function aLSTM:_accGradParameters(scale)

	if self._gLOutput then
		scale = scale or 1
		if self.batchsize then
			self._gLCell = self._gLCell:sum(1)
			self._gLCell:resize(self.outputSize)
			self._gLOutput = self._gLOutput:sum(1)
			self._gLOutput:resize(self.outputSize)
		end
		self.sbm.gradBias:narrow(1,1,self.outputSize):add(scale, self._gLCell)
		self.sbm.gradBias:narrow(1,self.fgstartid,self.outputSize):add(scale, self._gLOutput)
	end

end

-- init storage for tensor
function aLSTM:_tensor_clearState(tsr)

	tsr = tsr or self.sbm.bias

	-- cell sequence
	if not self._cell then
		self._cell = tsr.new()
	else
		self._cell:resize(0)
	end

	-- last cell
	self.cell = nil

	-- output sequence
	if not self.output then
		self.output = tsr.new()
	else
		self.output:resize(0)
	end

	-- last output
	-- here switch the usage of self.output and self._output for fit the standard of nn.Module
	-- just point self._output to keep aLSTM standard
	self._output = self.output

	-- gradInput sequence
	if not self.gradInput then
		self.gradInput = tsr.new()
	else
		self.gradInput:resize(0)
	end

	-- after tanh value for the final output from cell
	if not self.otanh then
		self.otanh = tsr.new()
	else
		self.otanh:resize(0)
	end

	-- output of the input and forget gate
	if not self.oifgate then
		self.oifgate = tsr.new()
	else
		self.oifgate:resize(0)
	end

	-- output of the output gate
	if not self.oogate then
		self.oogate = tsr.new()
	else
		self.oogate:resize(0)
	end

	-- output of z(hidden)
	if not self.ohid then
		self.ohid = tsr.new()
	else
		self.ohid:resize(0)
	end

	-- grad from the sequence after
	self._gLCell = nil
	self._gLOutput = nil

end

-- clear the storage
function aLSTM:_table_clearState()

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
	-- output of the input and forget gate
	self.oifgate = {}
	-- output of the output gate
	self.oogate = {}
	-- output of z(hidden)
	self.ohid = {}

	-- grad from the sequence after
	self._gLCell = nil
	self._gLOutput = nil

end

-- forget the history
function aLSTM:forget()

	self:clearState()

	for _, module in ipairs(self.modules) do
		module:forget()
	end

	-- clear last cell and output
	self.lastCell = nil
	self.lastOutput = nil

	-- set first sequence(will update bias)
	self.firstSequence = true

end

-- define type
function aLSTM:type(type, ...)

	return parent.type(self, type, ...)

end

-- reset the module
function aLSTM:reset(stdv)

	self.ifgate = self:buildIFModule()
	self.zmod = self:buildUpdateModule()
	self.ogate = self:buildOGModule()

	-- inner parameters need to correctly processed
	-- in fact, it is output and cell at time step 0
	-- it contains by a module to fit Container
	self.sbm = self:buildSelfBias(self.outputSize)

	-- module used to compute the forward and backward of tanh
	-- It seems does not need to be put in self.modules
	self.tanh = nn.aTanh(true)

	--[[ put the modules in self.modules,
	so the default method could be done correctly]]
	self.modules = {self.ifgate, self.zmod, self.ogate, self.sbm, self.tanh}

	if stdv then
		self:_ApplyReset(stdv)
	end

	self:forget()

end

function aLSTM:buildIFModule()

	local _ifm = nn.aSequential()
		:add(nn.aConcatTable()
			:add(nn.aSequential()
				:add(nn.aNarrowTable(1,2))
				:add(nn.aJoinTable(self.narrowDim, self.narrowDim, true))
				:add(nn.aLinear(self.inputSize + self.outputSize, self.outputSize * 2)))
			:add(nn.aSequential()
				:add(nn.aSelectTable(-1))
				:add(nn.aRepeat(2, self.narrowDim, self.narrowDim))
				--[[:add(nn.aConcatTable()
					:add(nn.Identity())
					:add(nn.Identity()))
				:add(nn.aJoinTable(self.narrowDim, self.narrowDim))]]
				:add(nn.aCMul(self.outputSize * 2))))
		:add(nn.aCAddTable())
		:add(nn.aSigmoid(true))

	return _ifm

end

-- build input and forget gate
--[[function aLSTM:buildIFModule()

	local _ifm = nn.aSequential()
		:add(nn.aJoinTable(self.narrowDim, self.narrowDim, true))
		:add(nn.aLinear(self.inputSize + self.outputSize * 2, self.outputSize * 2))
		:add(nn.aSigmoid(true))

	return _ifm

end]]

-- build output gate
function aLSTM:buildOGModule()

	local _ogm = nn.aSequential()
		:add(nn.aConcatTable()
			:add(nn.aSequential()
				:add(nn.aNarrowTable(1,2))
				:add(nn.aJoinTable(self.narrowDim, self.narrowDim, true))
				:add(nn.aLinear(self.inputSize + self.outputSize, self.outputSize)))
			:add(nn.aSequential()
				:add(nn.aSelectTable(-1))
				:add(nn.aCMul(self.outputSize))))
		:add(nn.aCAddTable())
		:add(nn.aSigmoid(true))

	return _ogm

end

-- build output gate
--[[function aLSTM:buildOGModule()

	local _ogm = nn.aSequential()
		:add(nn.aJoinTable(self.narrowDim, self.narrowDim, true))
		:add(nn.aLinear(self.inputSize + self.outputSize * 2, self.outputSize))
		:add(nn.aSigmoid(true))

	return _ogm

end]]

-- build z(update) module
function aLSTM:buildUpdateModule()

	local _um = nn.aSequential()
		:add(nn.aJoinTable(self.narrowDim, self.narrowDim, true))
		:add(nn.aLinear(self.inputSize + self.outputSize, self.outputSize))
		:add(nn.aTanh(true))

	return _um

end

-- build a module that contains aLSTM.bias and aLSTM.gradBias to make it fit Container
function aLSTM:buildSelfBias(outputSize)

	local _smb = nn.Module()
	_smb.bias = torch.zeros(2 * outputSize)
	_smb.gradBias = _smb.bias:clone()

	return _smb

end

-- prepare for LSTM
--[==[function aLSTM:prepare()

	-- Warning: This method may be DEPRECATED at any time
	-- it is for debug use
	-- you should write a faster and simpler module instead of nn
	-- for your particular use

	nn.aLinear = nn.Linear

	-- Warning: Use Sequence Tanh and Sigmoid are fast
	-- but be very very cautious!!!
	-- you need to give it an argument true,
	-- if you need it work in reverse order
	-- and you must turn to evaluate state if you were evaluate,
	-- otherwise the output are remembered!
	require "aSeqTanh"
	nn.aTanh = nn.aSeqTanh
	--nn.aTanh = nn.Tanh
	require "aSeqSigmoid"
	nn.aSigmoid = nn.aSeqSigmoid
	require "aSeqJoinTable"
	nn.aJoinTable = nn.aSeqJoinTable
	--nn.aJoinTable = nn.JoinTable
	--nn.aSigmoid = nn.Sigmoid
	--[[require "aSTTanh"
	require "aSTSigmoid"
	nn.aTanh = nn.aSTTanh
	nn.aSigmoid = nn.aSTSigmoid]]
	nn.aSequential = nn.Sequential

end]==]

-- introduce self
function aLSTM:__tostring__()

	return string.format('%s(%d -> %d)', torch.type(self), self.inputSize, self.outputSize)

end
