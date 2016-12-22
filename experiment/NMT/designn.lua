--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts defines the NMT model to train

	Version 0.0.3

]]

require "dpnn"
require "rnn"
require "nngraph"
require "SeqDropout"
require "vecLookup"
require "maskZerovecLookup"
require "ASequencerCriterion"

require "nngraph"
require "dep.gmodule"
require "dep.Sequencer"

require "prepare_module"

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
	local attention = nn.Recursor(nn.aTtention(sizvec, true))
	local classifier = nn.MaskZero(nn.NCEModule(sizvec * 2, nclass, knegsample, unigrams), 1)
	local NMT = nn.aGlobalAttentionNMT(encoder, decoder, attention, classifier, eosid, nil, true)
	local transfer = nn.aTransfer(eosid)
	local nnmod = buildNMT(id2vec,NMT,nn.Sequencer(classifier),nn.Sequencer(nn.Sequential():add(nn.JoinTable(2,2)):add(nn.Dropout(pDropout,nil,nil,true))),transfer)--[[nn.Sequential()
		:add(nn.ConcatTable()
			:add(nn.Sequential()
				:add(nn.ParallelTable()
					:add(id2vec)
					:add(nn.Identity()))
				:add(NMT)
				:add(nn.Sequencer(nn.JoinTable(2,2))))
			:add(nn.SelectTable(-1)))
		:add(nn.ZipTable())
		:add(nn.Sequencer(classifier))]]
	
	return nnmod
end

function buildNMT(id2vec,NMT,classifier,pNMT,transfer)
	local i1=id2vec()
	local i2=nn.Identity()()
	local oNMT=NMT({i1,i2})
	local poNMT=pNMT(oNMT)
	local zo=nn.ZipTable()({poNMT,transfer(i2)})
	local output = classifier(zo)
	return nn.gModule({i1,i2}, {output})
end

function buildEncoder()

	local nnmod = nn.Sequential()
		:add(nn.SeqDropout(pDropout))
		--[[:add(nn.ConcatTable()
			:add(nn.Narrow(1,1,-2))
			:add(nn.Narrow(1,2,-1)))
		:add(nn.JoinTable(2,2))
		:add(nn.SplitTable(1))]]
	local inputSize = sizvec-- * 2
	for _, hsize in ipairs(hiddenSize) do
		nnmod:add(nn.aSeqGRU(inputSize, hsize, true))
		nnmod:add(nn.Sequencer(nn.NormStabilizer()))
		inputSize = hsize
	end
	return nnmod

end

function buildDecoder()

	local nnmod = nn.Sequential()
		:add(nn.JoinTable(2,2))
		--:add(nn.Dropout(pDropout,nil,nil,true))
	local inputSize = sizvec * 2
	for _, hsize in ipairs(hiddenSize) do
		nnmod:add(nn.aStepGRU(inputSize, hsize, true))
		nnmod:add(nn.NormStabilizer())
		inputSize = hsize
	end
	--nnmod:add(nn.Dropout(pDropout,nil,nil,true))
	return nnmod

end

function getcrit()
	return nn.ASequencerCriterion(nn.MaskZeroCriterion(nn.NCECriterion(),0));
end

function setupvec(modin,value)
	--modin:get(1):get(1):get(1):get(1).updatevec = value
end

function dupvec(modin)
	setupvec(modin,false)
end

function upvec(modin)
	setupvec(modin,true)
end

function setnormvec(modin,value)
	--modin:get(1):get(1):get(1):get(1).usenorm = value
end

function dnormvec(modin)
	setnormvec(modin,false)
end

function normvec(modin)
	setnormvec(modin,true)
end