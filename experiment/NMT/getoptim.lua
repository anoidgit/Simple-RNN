--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts define getoptim, returns a optimization method

	Version 0.0.1

]]

function getoptim()
	local optim = require "optim"
	return optim.adam
end
