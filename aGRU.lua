--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement a standard GRU:

	z[t] = σ(W[x->z]x[t] + W[s->z]s[t−1] + b[1->z])            (1)
	r[t] = σ(W[x->r]x[t] + W[s->r]s[t−1] + b[1->r])            (2)
	h[t] = tanh(W[x->h]x[t] + W[hr->c](s[t−1]r[t]) + b[1->h])  (3)
	s[t] = (1-z[t])h[t] + z[t]s[t-1]                           (4)

	Version 0.2.6

]]

local aGRU, parent = torch.class('nn.aGRU', 'nn.aBstractSeq')
--local aGRU, parent = torch.class('nn.aGRU', 'nn.aBstractStep')

-- generate a module
function aGRU:__init(inputSize, outputSize, maskZero, remember, needcopyForward, needcopyBackward, narrowDim)

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

	self.narrowDim = narrowDim or 2

	self.needcopyForward = needcopyForward
	self.needcopyBackward = needcopyBackward

	self:reset()-- enable this for stdbuild
	--self:reset(1.0 / math.sqrt(3 * outputSize))

end

-- asign default method
--[[function aGRU:_asign(seqd)

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

function aGRU:_Copy(data, isForwardCopy)

	if isForwardCopy then
		self.output = data[1]
	else
		self._gLOutput = data[1]
		self.backwardCopied = true
	end

end

-- prepare data for the first step
function aGRU:_prepare_data(input)

	-- set batch size and prepare the output
	local _nIdim = input:nDimension()
	if _nIdim > 1 then

		self.batchsize = input:size(1)

		if self.rememberState and self.lastOutput then
			if self.lastOutput:size(1) == self.batchsize then
				self.output0 = self.lastOutput
			else
				self.output0 = self.lastOutput:narrow(1, 1, self.batchsize)
			end
		else
			self.output0 = self.sbm.bias
			self.output0 = self.output0:reshape(1,self.outputSize):expand(self.batchsize, self.outputSize)
		end

		-- narrow dimension
		self.narrowDim = _nIdim

	else

		self.batchsize = nil

		if self.rememberState and self.lastOutput then
			self.output0 = self.lastOutput
		else
			self.output0 = self.sbm.bias
		end

		-- narrow dimension
		self.narrowDim = 1

	end

end

-- updateOutput called by forward,
-- It input a time step's input and produce an output
function aGRU:_step_updateOutput(input)

	-- ensure output are ready for the first step
	if not self.output or self.output:size(1)~=input:size(1) then

		self:_prepare_data(input)

		self.output = self.output0
	end

	-- compute input gate and forget gate
	local _ifgo = self.ifgate:forward({input, self.output})

	-- get input gate and forget gate
	local _igo = _ifgo:narrow(self.narrowDim, 1, self.outputSize)
	local _fgo = _ifgo:narrow(self.narrowDim, self.fgstartid, self.outputSize)

	-- compute reset output
	local _ro = torch.cmul(self.output, _fgo)

	-- compute hidden
	local _ho = self.hmod:forward({input, _ro})

	local _igr = _igo.new()
	_igr:resizeAs(_igo):fill(1):csub(_igo)

	-- compute the final output for this input
	self.output = torch.cmul(_igr, _ho) + torch.cmul(_igo, self.output)

	-- if training, remember what should remember
	if self.train then
		table.insert(self.outputs, self.output)--h[t]
		table.insert(self.oifgate, _ifgo)--if[t], input and forget
		table.insert(self.ohid, _ho)-- h[t]
		table.insert(self.oigr, _igr)-- 1-z[t]
		table.insert(self.ro, _ro)-- reset output
	end

	if self.needcopyForward then
		self.memTCopy = {self.output}
	end

	-- return the output for this input
	return self.output

end

-- updateOutput for tensor input,
-- input tensor is expected to be seqlen * batchsize * vecsize
function aGRU:_tseq_updateOutput(input)

	self.gradInput:resize(0)

	-- get input and output size
	local iSize = input:size()
	local oSize = input:size()
	oSize[3] = self.outputSize
	self.output = input.new()
	self.output:resize(oSize)

	if self.train then

		self.oigr:resize(oSize)
		self.ro:resize(oSize)
		self.ohid:resize(oSize)
		local dOSize = input:size()
		dOSize[3] = self.outputSize * 2
		self.oifgate:resize(dOSize)

	end

	self:_prepare_data(input[1])

	local _output = self.output0

	-- forward the whole sequence
	for _t = 1, iSize[1] do
		local iv = input[_t]
		-- compute input gate and forget gate
		local _ifgo = self.ifgate:forward({iv, _output})

		-- get input gate and forget gate
		local _igo = _ifgo:narrow(self.narrowDim, 1, self.outputSize)
		local _fgo = _ifgo:narrow(self.narrowDim, self.fgstartid, self.outputSize)

		-- compute reset output
		local _ro = torch.cmul(_output, _fgo)

		-- compute hidden
		local _ho = self.hmod:forward({iv, _ro})

		local _igr = _igo.new()
		_igr:resizeAs(_igo):fill(1):csub(_igo)

		-- compute the final output for this input
		_output = torch.cmul(_igr, _ho) + torch.cmul(_igo, _output)

		self.output[_t]:copy(_output)

		-- if training, remember what should remember
		if self.train then
			self.oifgate[_t]:copy(_ifgo)--if[t], input and forget
			self.ohid[_t]:copy(_ho)-- h[t]
			self.oigr[_t]:copy(_igr)-- 1-z[t]
			self.ro[_t]:copy(_ro)-- reset output
		end

	end

	if self.needcopyForward then
		self.memTCopy = {_output}
	end

	return self.output

end

-- updateOutput for a table sequence
function aGRU:_seq_updateOutput(input)

	self.gradInput = nil

	local output = {}

	self:_prepare_data(input[1])

	local _output = self.output0

	-- forward the whole sequence
	for _,iv in ipairs(input) do
		-- compute input gate and forget gate
		local _ifgo = self.ifgate:forward({iv, _output})

		-- get input gate and forget gate
		local _igo = _ifgo:narrow(self.narrowDim, 1, self.outputSize)
		local _fgo = _ifgo:narrow(self.narrowDim, self.fgstartid, self.outputSize)

		-- compute reset output
		local _ro = torch.cmul(_output, _fgo)

		-- compute hidden
		local _ho = self.hmod:forward({iv, _ro})

		local _igr = _igo.new()-- 1-z[t]
		_igr:resizeAs(_igo):fill(1):csub(_igo)

		-- compute the final output for this input
		_output = torch.cmul(_igr, _ho) + torch.cmul(_igo, _output)

		table.insert(output, _output)

		-- if training, remember what should remember
		if self.train then
			--table.insert(self.outputs, _output)--h[t]
			table.insert(self.oifgate, _ifgo)--if[t], input and forget
			table.insert(self.ohid, _ho)-- h[t]
			table.insert(self.oigr, _igr)-- 1-z[t]
			table.insert(self.ro, _ro)-- reset output
		end

	end

	if self.needcopyForward then
		self.memTCopy = {_output}
	end

	--[[for _,v in ipairs(input) do
		table.insert(output,self:_step_updateOutput(v))
	end]]

	-- this have conflict with _step_updateOutput,
	-- but anyhow do not use them at the same time
	self.output = output
	self.outputs = output

	--[[if self.train then
		self:_check_table_same(self._cell)
		self:_check_table_same(self.outputs)
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
--[[function aGRU:_check_table_same(tbin)

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
function aGRU:_step_backward(input, gradOutput, scale)

	-- if need to mask zero, then mask
	if self.maskzero then
		self:_step_makeZero(input, gradOutput)
	end

	local gradInput

	local nfirstep = true

	-- temp storage
	local __gInput

	-- pre claim the local variable, they were discribed where they were used.
	local _cPrevOutput, _coifgate, _coh, _coigate, _cofgate, _gg, _coigr, _cro, _gro

	-- if this is not the last step
	if self._gLOutput then

		-- add gradOutput from the sequence behind
		gradOutput:add(self._gLOutput)

		-- if self._gLOutput was copied from later,
		-- then process the last output
		if self.backwardCopied then

			-- remove the last output, because it was never used
			local _lastOutput = table.remove(self.outputs)

			-- remember the end of sequence for next input use
			if self.rememberState then
				self.lastOutput = _lastOutput
			end

			self.backwardCopied = nil

		end

	else

		-- remove the last output, because it was never used
		local _lastOutput = table.remove(self.outputs)
		
		-- remember the end of sequence for next input use
		if self.rememberState then
			self.lastOutput = _lastOutput
		end

	end

	if #self.outputs > 0 then
		_cPrevOutput = table.remove(self.outputs)-- previous output
	else
		_cPrevOutput = self.output0
		nfirstep = false
	end

	_coh = table.remove(self.ohid)-- hidden unit produced by input, h[t]
	_coigr = table.remove(self.oigr)-- 1-z[t]
	_cro = table.remove(self.ro)-- reset output
	_coifgate = table.remove(self.oifgate)-- output of the input and forget gate, if[t], input and forget

	-- asign output of input gate and output gate
	_coigate = _coifgate:narrow(self.narrowDim, 1, self.outputSize)-- i[t]
	_cofgate = _coifgate:narrow(self.narrowDim, self.fgstartid, self.outputSize)--f[t] 

	-- backward

	-- backward hidden
	gradInput, _gro = unpack(self.hmod:backward({input, _cro}, torch.cmul(gradOutput, _coigr), scale))-- gradient on input and reset of previous output

	-- backward ifgate(input and forget gate)
	-- compute gradOutput
	_gg = gradInput.new()
	if self.batchsize then
		_gg:resize(self.batchsize, 2 * self.outputSize)
	else
		_gg:resize(2 * self.outputSize)
	end
	_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(gradOutput, _cPrevOutput - _coh))
	_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gro, _cPrevOutput))
	-- backward the gate

	__gInput, self._gLOutput = unpack(self.ifgate:backward({input, _cPrevOutput}, _gg, scale))
	gradInput:add(__gInput)

	if nfirstep then

		self._gLOutput:add(torch.cmul(_gro, _cPrevOutput))
		self._gLOutput:add(torch.cmul(gradOutput, _coigate))

	else

		-- accGradParameters for self
		if self.rememberState then

			if self.firstSequence then

				self._gLOutput:add(torch.cmul(_gro, _cPrevOutput))
				self._gLOutput:add(torch.cmul(gradOutput, _coigate))
				-- accGradParameters for self
				self:_accGradParameters(scale)
				self.firstSequence = false

			end

			-- prepare for next forward sequence, clear cache
			self:clearState()

		else

			self._gLOutput:add(torch.cmul(_gro, _cPrevOutput))
			self._gLOutput:add(torch.cmul(gradOutput, _coigate))
			self:_accGradParameters(scale)

			self:forget()

		end

	end

	-- this have conflict with _table_seq_backward,
	-- but anyhow do not use them at the same time
	self.gradInput = gradInput:clone()

	return self.gradInput

end

-- backward process the whole sequence
-- it takes the whole input, gradOutput sequence as input
-- and it will clear the cache after done backward
function aGRU:_seq_backward(input, gradOutput, scale)

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
	local _lastOutput = table.remove(self.outputs)

	-- remember the end of sequence for next input use
	if self.rememberState then
		self.lastOutput = _lastOutput
	end

	-- grad to input
	local _gInput

	-- temp storage
	local __gInput

	-- pre claim the local variable, they were discribed where they were used.
	local _cGradOut, _cInput, _cPrevOutput, _coifgate, _coh, _coigate, _cofgate, _gg, _coigr, _cro, _gro

	-- backward the seq
	for _t = _length, 1, -1 do

		-- prepare data for future use
		_cGradOut = table.remove(gradOutput)-- current gradOutput

		if self._gLOutput then
			-- add gradOutput from the sequence behind
			_cGradOut:add(self._gLOutput)
		end

		_cInput = table.remove(_input)-- current input

		if _t > 1 then
			_cPrevOutput = table.remove(self.outputs)-- previous output, s[t-1]
		else
			_cPrevOutput = self.output0
		end

		_coh = table.remove(self.ohid)-- hidden unit produced by input, h[t]
		_coigr = table.remove(self.oigr)-- 1-z[t]
		_cro = table.remove(self.ro)-- reset output
		_coifgate = table.remove(self.oifgate)-- output of the input and forget gate, if[t], input and forget

		-- asign output of input gate and output gate
		_coigate = _coifgate:narrow(self.narrowDim, 1, self.outputSize)-- i[t]
		_cofgate = _coifgate:narrow(self.narrowDim, self.fgstartid, self.outputSize)--f[t] 

		-- backward

		-- backward hidden
		_gInput, _gro = unpack(self.hmod:backward({_cInput, _cro}, torch.cmul(_cGradOut, _coigr), scale))-- gradient on input and reset of previous output

		-- backward ifgate(input and forget gate)
		-- compute gradOutput
		_gg = _gInput.new()
		if self.batchsize then
			_gg:resize(self.batchsize, 2 * self.outputSize)
		else
			_gg:resize(2 * self.outputSize)
		end
		_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_cGradOut, _cPrevOutput - _coh))
		_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gro, _cPrevOutput))
		-- backward the gate

		__gInput, self._gLOutput = unpack(self.ifgate:backward({_cInput, _cPrevOutput}, _gg, scale))
		_gInput:add(__gInput)

		if _t > 1 then

			self._gLOutput:add(torch.cmul(_gro, _cPrevOutput))
			self._gLOutput:add(torch.cmul(_cGradOut, _coigate))
			gradInput[_t] = _gInput:clone()

		else

			gradInput[_t] = _gInput

		end

	end

	-- accGradParameters for self
	if self.rememberState then

		if self.firstSequence then

			self._gLOutput:add(torch.cmul(_gro, _cPrevOutput))
			self._gLOutput:add(torch.cmul(_cGradOut, _coigate))
			-- accGradParameters for self
			self:_accGradParameters(scale)
			self.firstSequence = false

		end

		-- prepare for next forward sequence, clear cache
		self:clearState()

	else

		self._gLOutput:add(torch.cmul(_gro, _cPrevOutput))
		self._gLOutput:add(torch.cmul(_cGradOut, _coigate))
		self:_accGradParameters(scale)

		self:forget()

	end

	self.gradInput = gradInput

	return self.gradInput

end

-- backward for tensor input and gradOutput sequence
function aGRU:_tseq_backward(input, gradOutput, scale)

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
	-- get current cell

	-- remember the end of sequence for next input use
	if self.rememberState then
		-- clone it, for fear that self.lastCell and self.lastOutput marks the whole memory of self.cell and self.output as used
		self.lastOutput = _lastOutput:clone()
	end

	-- grad to input
	local _gInput

	-- temp storage
	local __gInput

	-- pre claim the local variable, they were discribed where they were used.
	local _cGradOut, _cInput, _cPrevOutput, _coifgate, _coh, _coigate, _cofgate, _gg, _coigr, _cro, _gro

	-- backward the seq
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
		else
			_cPrevOutput = self.output0
		end

		_coh = self.ohid[_t]-- hidden unit produced by input, h[t]
		_coigr = self.oigr[_t]-- 1-z[t]
		_cro = self.ro[_t]-- reset output
		_coifgate = self.oifgate[_t]-- output of the input and forget gate, if[t], input and forget

		-- asign output of input gate and output gate
		_coigate = _coifgate:narrow(self.narrowDim, 1, self.outputSize)-- i[t]
		_cofgate = _coifgate:narrow(self.narrowDim, self.fgstartid, self.outputSize)--f[t] 

		-- backward

		-- backward hidden
		_gInput, _gro = unpack(self.hmod:backward({_cInput, _cro}, torch.cmul(_cGradOut, _coigr), scale))-- gradient on input and reset of previous output

		-- backward ifgate(input and forget gate)
		-- compute gradOutput
		_gg = _gInput.new()
		if self.batchsize then
			_gg:resize(self.batchsize, 2 * self.outputSize)
		else
			_gg:resize(2 * self.outputSize)
		end
		_gg:narrow(self.narrowDim, 1, self.outputSize):copy(torch.cmul(_cGradOut, _cPrevOutput - _coh))
		_gg:narrow(self.narrowDim, self.fgstartid, self.outputSize):copy(torch.cmul(_gro, _cPrevOutput))
		-- backward the gate

		__gInput, self._gLOutput = unpack(self.ifgate:backward({_cInput, _cPrevOutput}, _gg, scale))
		_gInput:add(__gInput)

		if _t > 1 then

			self._gLOutput:add(torch.cmul(_gro, _cPrevOutput))
			self._gLOutput:add(torch.cmul(_cGradOut, _coigate))

		end

		gradInput[_t]:copy(_gInput)

	end

	-- accGradParameters for self
	if self.rememberState then

		if self.firstSequence then

			self._gLOutput:add(torch.cmul(_gro, _cPrevOutput))
			self._gLOutput:add(torch.cmul(_cGradOut, _coigate))
			-- accGradParameters for self
			self:_accGradParameters(scale)
			self.firstSequence = false

		end

		-- prepare for next forward sequence, clear cache
		self:clearState()

	else

		self._gLOutput:add(torch.cmul(_gro, _cPrevOutput))
		self._gLOutput:add(torch.cmul(_cGradOut, _coigate))
		self:_accGradParameters(scale)

		self:forget()

	end

	self.gradInput:resizeAs(gradInput):copy(gradInput)

	return self.gradInput

end

-- updateParameters 
--[[function aGRU:updateParameters(learningRate)

	for _, module in ipairs(self.modules) do
		module:updateParameters(learningRate)
	end
	self.sbm.bias:add(-learningRate, self.sbm.gradBias)

end]]

-- zeroGradParameters
--[[function aGRU:zeroGradParameters()

	for _, module in ipairs(self.modules) do
		module:zeroGradParameters()
	end
	self.sbm.gradBias:zero()

end]]

-- accGradParameters used for aGRU.bias
function aGRU:_accGradParameters(scale)

	if self._gLOutput then

		if self.needcopyBackward then
			self.gradTCopy = {self._gLOutput:clone()}
		end

		scale = scale or 1
		if self.batchsize then
			self._gLOutput = self._gLOutput:sum(1)
			self._gLOutput:resize(self.outputSize)
		end
		self.sbm.gradBias:add(scale, self._gLOutput)
	end

end

-- init storage for tensor
function aGRU:_tensor_clearState(tsr)

	tsr = tsr or self.sbm.bias

	-- output sequence
	if not self.output then
		self.output = tsr.new()
	else
		self.output:resize(0)
	end

	-- last output
	-- here switch the usage of self.output and self.outputs for fit the standard of nn.Module
	-- just point self.outputs to keep aGRU standard
	self.outputs = self.output

	-- gradInput sequence
	if not self.gradInput then
		self.gradInput = tsr.new()
	else
		self.gradInput:resize(0)
	end

	-- output of the input and forget gate
	if not self.oifgate then
		self.oifgate = tsr.new()
	else
		self.oifgate:resize(0)
	end

	-- output of z(hidden)
	if not self.ohid then
		self.ohid = tsr.new()
	else
		self.ohid:resize(0)
	end

	-- 1-z[t]
	if not self.oigr then
		self.oigr = tsr.new()
	else
		self.oigr:resize(0)
	end

	-- reset output
	if not self.ro then
		self.ro = tsr.new()
	else
		self.ro:resize(0)
	end

	-- grad from the sequence after
	self._gLOutput = nil

	-- if true, _step_backward will know it need to process the final output,
	-- even self.__gLOutput is not nil
	self.backwardCopied = nil

end

-- clear the storage
function aGRU:_table_clearState()

	-- output sequence
	self.outputs = {}
	-- last output
	self.output = nil
	-- gradInput sequence
	self.gradInput = nil

	-- output of the input and forget gate
	self.oifgate = {}
	-- output of z(hidden)
	self.ohid = {}
	-- 1-z[t]
	self.oigr = {}
	-- reset output
	self.ro = {}

	-- grad from the sequence after
	self._gLOutput = nil

	-- if true, _step_backward will know it need to process the final output,
	-- even self.__gLOutput is not nil
	self.backwardCopied = nil

end

-- forget the history
function aGRU:forget()

	self:clearState()

	for _, module in ipairs(self.modules) do
		module:forget()
	end

	-- clear last output
	self.lastOutput = nil

	-- set first sequence(will update bias)
	self.firstSequence = true

end

-- define type
function aGRU:type(type, ...)

	return parent.type(self, type, ...)

end

-- reset the module
function aGRU:reset(stdv)

	self.ifgate = self:buildIFModule()
	self.hmod = self:buildUpdateModule()

	-- inner parameters need to correctly processed
	-- in fact, it is output and cell at time step 0
	-- it contains by a module to fit Container
	self.sbm = self:buildSelfBias(self.outputSize)

	--[[ put the modules in self.modules,
	so the default method could be done correctly]]
	self.modules = {self.ifgate, self.hmod, self.sbm}

	if stdv then
		self:_ApplyReset(stdv)
	end

	self:forget()

end

-- build input and forget gate
function aGRU:buildIFModule(stdbuild)

	local _ifm
	if stdbuild then
		_ifm = nn.Sequential()
			:add(nn.aConcatTable()
				:add(nn.aSequential()
					:add(nn.aParallelTable()
						:add(nn.aLinear(self.inputSize, self.outputSize))
						:add(nn.aLinear(self.outputSize, self.outputSize, false)))
					:add(nn.aCAddTable()))
				:add(nn.aSequential()
					:add(nn.aParallelTable()
						:add(nn.aLinear(self.inputSize, self.outputSize))
						:add(nn.aLinear(self.outputSize, self.outputSize, false)))
					:add(nn.aCAddTable())))
			:add(nn.aJoinTable(self.narrowDim, self.narrowDim))
			:add(nn.aSigmoid())

	else
		_ifm = nn.aSequential()
			:add(nn.aJoinTable(self.narrowDim,self.narrowDim))
			:add(nn.aLinear(self.inputSize + self.outputSize, self.outputSize * 2))
			:add(nn.aSigmoid())
	end

	return nn.Recursor(_ifm)

end

-- build z(update) module
function aGRU:buildUpdateModule()

	local _um = nn.aSequential()
		:add(nn.aJoinTable(self.narrowDim, self.narrowDim))
		:add(nn.aLinear(self.inputSize + self.outputSize, self.outputSize))
		:add(nn.aTanh())

	return nn.Recursor(_um)

end

-- build a module that contains aGRU.bias and aGRU.gradBias to make it fit Container
function aGRU:buildSelfBias(outputSize)

	local _smb = nn.Module()
	_smb.bias = torch.zeros(outputSize)
	_smb.gradBias = _smb.bias:clone()

	return _smb

end

-- prepare for GRU
--[=[function aGRU:prepare()

	-- Warning: This method may be DEPRECATED at any time
	-- it is for debug use
	-- you should write a faster and simpler module instead of nn
	-- for your particular use

	nn.aJoinTable = nn.JoinTable
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
	--nn.aSigmoid = nn.Sigmoid
	--[[require "aSTTanh"
	require "aSTSigmoid"
	nn.aTanh = nn.aSTTanh
	nn.aSigmoid = nn.aSTSigmoid]]
	nn.aSequential = nn.Sequential

end]=]

-- introduce self
function aGRU:__tostring__()

	return string.format('%s(%d -> %d)', torch.type(self), self.inputSize, self.outputSize)

end
