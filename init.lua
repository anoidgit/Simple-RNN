-- create global sarnn table:
sarnn = {}
sarnn.version = 1

unpack = unpack or table.unpack

-- support modules
torch.include('sarnn', 'dep/aSeqTanh.lua')
torch.include('sarnn', 'dep/aSeqSigmoid.lua')
torch.include('sarnn', 'dep/aSeqSoftMax.lua')
torch.include('sarnn', 'aBstractBase.lua')
torch.include('sarnn', 'aBstractSeq.lua')
torch.include('sarnn', 'aBstractStep.lua')
torch.include('sarnn', 'aGRU.lua')
torch.include('sarnn', 'aLSTM.lua')
torch.include('sarnn', 'dep/aSeqBiLinear.lua')
torch.include('sarnn', 'dep/aTtention.lua')

-- prevent likely name conflicts
nn.sarnn = sarnn
