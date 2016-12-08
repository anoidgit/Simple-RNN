--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts defines the NMT model to train

	Version 0.0.3

]]

require "dpnn"
require "rnn"
require "SeqDropout"
require "vecLookup"
require "maskZerovecLookup"
require "ASequencerCriterion"

require "aSeqSigmoid"
require "aSeqTanh"
require "aBstractBase"
require "aBstractSeq"
require "aBstractStep"
require "aRepeat"
-- aSeqLSTM and aStepLSTM was just aLSTM derived from different parent class: aBstractSeq and aBstractStep
require "aSeqLSTM"
require "aStepLSTM"

nn.aTanh = nn.aSeqTanh
nn.aSigmoid = nn.aSeqSigmoid
nn.aLinear = nn.Linear
nn.aSequential = nn.Sequential
nn.aJoinTable = nn.JoinTable
nn.aNarrowTable = nn.NarrowTable
nn.aSelectTable = nn.SelectTable
nn.aCMul = nn.CMul
nn.aCAddTable = nn.CAddTable
nn.aConcatTable = nn.ConcatTable

require "aSeqBiLinearScore"
require "aTtention"

require "aBstractNMT"
require "aGlobalAttentionNMT"

function getnn()
	--return getonn()
	return getnnn()
end

function getonn()
	wvec = nil
	--local lmod = loadObject("modrs/nnmod.asc").module
	local lmod = torch.load("modrs/nnmod.asc").module
	return lmod
end

function getnnn()

	require "nnsettings"

	local id2vec = nn.maskZerovecLookup(wvec);
	wvec = nil

	local encoder = buildEncoder()
	local decoder = buildDecoder()
	local attention = nn.aTtention(sizvec, true)
	local classifier = nn.NCEModule(sizvec * 2, nclass, knegsample, unigrams)
	local NMT = nn.aGlobalAttentionNMT(encoder, decoder, attention, classifier)
	local nnmod = nn.Sequential()
		:add(nn.ConcatTable()
			:add(nn.Sequential()
				:add(nn.ParallelTable()
					:add(id2vec)
					:add(nn.Identity()))
				:add(NMT))
			:add(nn.SelectTable(-1)))
		:add(nn.ZipTable())
	
	return nnmod
end

function buildEncoder()

	local nnmod = nn.Sequential()
		:add(nn.SeqDropout(pDropout))
	local inputSize = sizvec
	for _, hsize in ipairs(hiddenSize) do
		nnmod:add(nn.aSeqLSTM(inputSize, hsize, true))
		nnmod:add(nn.Sequencer(nn.NormStabilizer()))
		inputSize = hsize
	end
	return nnmod

end

function buildDecoder()

	local nnmod = nn.Sequential()
		:add(nn.SeqDropout(pDropout))
	local inputSize = sizvec * 2
	for _, hsize in ipairs(hiddenSize) do
		nnmod:add(nn.aStepLSTM(inputSize, hsize, true))
		nnmod:add(nn.Sequencer(nn.NormStabilizer()))
		inputSize = hsize
	end
	return nnmod

end

function getcrit()
	return nn.ASequencerCriterion(nn.MaskZeroCriterion(nn.NCECriterion(),1));
end

function setupvec(modin,value)
	modin:get(1):get(1):get(1):get(1).updatevec = value
end

function dupvec(modin)
	setupvec(modin,false)
end

function upvec(modin)
	setupvec(modin,true)
end

function setnormvec(modin,value)
	modin:get(1):get(1):get(1):get(1).usenorm = value
end

function dnormvec(modin)
	setnormvec(modin,false)
end

function normvec(modin)
	setnormvec(modin,true)
end
