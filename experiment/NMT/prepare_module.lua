--[[require "dep.aSeqSigmoid"
require "dep.aSeqSoftMax"]]--
--require "dep.aSeqTanh"
require "dep.aTSoftMax"
require "dep.aBstractBase"
require "dep.aBstractSeq"
require "dep.aBstractStep"

--[[nn.aSigmoid = nn.aSeqSigmoid
nn.aSoftMax = nn.aSeqSoftMax]]--
nn.aTanh = nn.Tanh
nn.aSigmoid = nn.Sigmoid
nn.aLinear = nn.Linear
nn.aSequential = nn.Sequential
nn.aConcatTable = nn.ConcatTable
nn.aParallelTable = nn.ParallelTable
nn.aCAddTable = nn.CAddTable
nn.aJoinTable = nn.JoinTable

require "module.aTransfer"

require "module.aSeqGRU"
require "module.aStepGRU"

require "dep.aBiLinearScore"
require "module.aTtention"

require "dep.aBstractNMT"
require "module.aGlobalAttentionNMT"
