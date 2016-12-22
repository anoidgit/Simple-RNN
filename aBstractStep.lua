--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts implement an aBstractStep for RNN:

	Version 0.0.4

]]

local aBstractStep, parent = torch.class('nn.aBstractStep', 'nn.aBstractBase')

-- generate a module
function aBstractStep:__init()

	parent.__init(self)

end

-- fit torch standard,
-- calc an output for an input,
-- faster while you just specify which function to use
function aBstractStep:updateOutput(input)

	return self:_step_updateOutput(input)

end

-- fit torch standard,
-- backward,
-- faster while you just specify which function to use
function aBstractStep:backward(input, gradOutput, scale)

	return self:_step_backward(input, gradOutput, scale)

end

-- fit torch rnn standard,
-- clear the cache used
function aBstractStep:clearState()

	self:_table_clearState()

	--[[for _, module in ipairs(self.modules) do
		module:clearState()
	end]]

end

-- mask zero for a step
function aBstractStep:_step_makeZero(input, gradOutput)

	if self.batchsize then
		-- if batch input
		
		-- get a zero unit
		local _stdZero = input.new()
		_stdZero:resizeAs(input[1]):zero()
		-- look at each unit
		for _t = 1, self.batchsize do
			if input[_t]:equal(_stdZero) then
				-- if it was zero, then zero the gradOutput
				gradOutput[_t]:zero()
			end
		end

	else
		-- if not batch

		local _stdZero = input.new()
		_stdZero:resizeAs(input):zero()
		if input:equal(_stdZero) then
			gradOutput:zero()
		end
	end

end
