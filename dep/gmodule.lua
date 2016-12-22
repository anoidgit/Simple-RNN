local gModule = nn.gModule

local nesting = require('nngraph.nesting')
local utils = require('nngraph.utils')
local istensor = torch.isTensor
local istable = utils.istable
local istorchclass = utils.istorchclass

local function getTotalGradOutput(node)
	local gradOutput = node.data.gradOutput
	assert(istable(gradOutput), "expecting gradients to sum")
	if #gradOutput > 1 then
		-- Check if we can bypass the allocation, for the special case where all
		-- gradOutputs but one are zero tensors with an underlying one-element
		-- storage. Note that for the case that we
		-- cannot bypass it, this check will only be performed once
		if not node.data.gradOutputBuffer then
			local count = 0
			local idx = 1
			-- Count how many gradOutput are tensors of 1 element filled with zero
			for i=1,#gradOutput do
				local zero = torch.isTensor(gradOutput[i]) and
								 gradOutput[i]:storage() ~= nil and
								 gradOutput[i]:storage():size() == 1 and
								 gradOutput[i]:storage()[1] == 0
				if not zero then
					idx = i
					count = count + 1
				end
			end
			if count < 2 then
				-- Return the only non-zero one, or the first one
				-- if they are all zero
				return gradOutput[idx]
			end
		end
		node.data.gradOutputBuffer = node.data.gradOutputBuffer or nesting.cloneNested(gradOutput[1])
		local gobuff = node.data.gradOutputBuffer
		nesting.resizeNestedAs(gobuff, gradOutput[1])
		nesting.copyNested(gobuff, gradOutput[1])
		for i=2,#gradOutput do
			nesting.addNestedTo(gobuff, gradOutput[i])
		end
		gradOutput = gobuff
	else
		gradOutput = gradOutput[1]
	end
	return gradOutput
end

function gModule:backward(input,gradOutput,scale)
	local function neteval(node)
		if node.data.selectindex then
			assert(not node.data.module, "the selectindex-handling nodes should have no module")
			assert(#node.children == 1, "only the splitted node should be the input")
			local child = node.children[1]
			local go = getTotalGradOutput(node)
			child.data.gradOutput = child.data.gradOutput or {}
			assert(#child.data.gradOutput <= 1, "the splitted node should be used only once")
			-- The data.gradOutput holds the to-be-summed gradients.
			child.data.gradOutput[1] = child.data.gradOutput[1] or {}
			assert(not child.data.gradOutput[1][node.data.selectindex], "no gradOutput should be assigned yet")
			child.data.gradOutput[1][node.data.selectindex] = go
		else
			local gradOutput = getTotalGradOutput(node)
			-- backward through this node
			-- If no module is present, the node behaves like nn.Identity.
			local gradInput
			if not node.data.module then
				gradInput = gradOutput
			else
				local input = node.data.input
				-- a parameter node is captured
				if input == nil and node.data.module ~= nil then
					input = {}
				end
				if #input == 1 then
					input = input[1]
				end
				local module = node.data.module
				gradInput = module:backward(input,gradOutput,scale)
			end
			-- propagate the output to children
			for i,child in ipairs(node.children) do
				child.data.gradOutput = child.data.gradOutput or {}
				local mapindex = node.data.mapindex[child.data]
				local gi
				if #node.children == 1 then
					gi = gradInput
				else
					gi = gradInput[mapindex]
				end
				table.insert(child.data.gradOutput,gi)
			end
		end
		if self.verbose then
			print(' V : ' .. node:label())
		end
	end
	local outnode = self.outnode
	if #outnode.children > 1 and #gradOutput ~= #outnode.children then
		error(string.format('Got %s gradOutputs instead of %s', #gradOutput, #outnode.children))
	end
	for _,node in ipairs(self.backwardnodes) do
		local gradOutput = node.data.gradOutput
		while gradOutput and #gradOutput >0 do
			table.remove(gradOutput)
		end
	end
	-- Set the starting gradOutput.
	outnode.data.gradOutput = outnode.data.gradOutput or {}
	outnode.data.gradOutput[1] = gradOutput

	for i,node in ipairs(self.backwardnodes) do
		neteval(node)
	end

	assert(#self.innode.data.gradOutput == 1, "expecting the innode to be used only once")
	self.gradInput = self.innode.data.gradOutput[1]
	return self.gradInput
end