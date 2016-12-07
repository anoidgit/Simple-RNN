--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement an aBstractBase for RNN:

	Version 0.1.1

]]

local aBstractBase, parent = torch.class('nn.aBstractBase', 'nn.Container')

-- generate a module
function aBstractBase:__init()

	parent.__init(self)

end

function aBstractBase:_clearTunnel(clear_who)

	if clear_who then
		if clear_who == "forward" then
			self.memTCopy = nil-- table to keep the newest memory like:output and cell etc
		else
			self.gradTCopy = nil-- table to keep gradient to the memory like:output and cell etc
		end
	else
		self.memTCopy = nil
		self.gradTCopy = nil
	end

end

-- updateGradInput for sequence,
-- in fact, it call backward
function aBstractBase:updateGradInput(input, gradOutput)

	return self:backward(input, gradOutput)

end

-- modules in aBstractBase.modules were done while backward
function aBstractBase:accGradParameters(input, gradOutput, scale)

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
function aBstractBase:evaluate()

	self.train = false

	for _, module in ipairs(self.modules) do
		module:evaluate()
	end

	self:forget()

	self:_clearTunnel()

end

-- train
function aBstractBase:training()

	self.train = true

	for _, module in ipairs(self.modules) do
		module:training()
	end

	self:forget()

	self:_clearTunnel()

end

-- remember last state or not
function aBstractBase:remember(mode)

	-- set default to both
	local _mode = mode or "both"

	if _mode == "both" or _mode == true then
		self.rememberState = true
	else
		self.rememberState = nil
	end

	self:forget()

end

-- init parameters
function aBstractBase:_ApplyReset(stdv)

	stdv = stdv or 1.0 / math.sqrt(self.outputSize + self.inputSize)

	for _, module in ipairs(self.modules) do
		module:reset(stdv)
	end

end

-- copy previous output and cell etc from encoder to decoder
function aBstractBase:_copy_forward(fromodel, tomodel)

	for _ = 1, fromodel:size() do
		local crfm = fromodel:get(_)
		if torch.isTypeOf(crfm, 'nn.aBstractBase') then
			local tarm = tomodel:get(_)
			tarm:_Copy(fromodel.memTCopy, true)
		end
	end

end

-- copy gradient to output and cell etc from decoder to encoder
function aBstractBase:_copy_backward(fromodel, tomodel)

	for _ = 1, fromodel:size() do
		local crfm = fromodel:get(_)
		if torch.isTypeOf(crfm, 'nn.aBstractBase') then
			local tarm = tomodel:get(_)
			tarm:_Copy(fromodel.gradTCopy)
		end
	end

end
