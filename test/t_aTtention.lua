print("set default tensor type")
torch.setdefaulttensortype('torch.FloatTensor')

seqlen = 2
batchsize = 4
vecsize = 4

require "nn"
require "aSeqBiLinearScore"
require "aTtention"
require "aSeqSoftMax"

nn.aLinear = nn.Linear
nn.aSequential = nn.Sequential
nn.aSoftMax = nn.aSeqSoftMax
--nn.aSoftMax = nn.SoftMax
--nn.aTranspose = nn.Transpose

tmod = nn.aTtention(vecsize, true)
tmod:training()
td1 = torch.randn(seqlen,batchsize,vecsize)
td2 = torch.randn(batchsize,vecsize)
ti = {td1, td2}
rs = tmod:forward(ti)
print(rs)
grad = tmod:backward(ti, torch.randn(batchsize*vecsize))
print(grad)
print(grad[1])
print(grad[2])
