print("set default tensor type")
torch.setdefaulttensortype('torch.FloatTensor')

require "cunn"
require "aLSTM"

isize=128
osize=isize
cycle=8
seqlen=4
bsize=8

tmod=nn.aLSTM(isize,osize,true):cuda()
tmod:training()

for _ = 1,cycle do
	td={}
	grad={}
	for __ = 1, seqlen do
		table.insert(td,torch.randn(bsize,isize):cuda())
		table.insert(grad,torch.randn(bsize,osize):cuda()/8192)
	end
	tmod:zeroGradParameters()
	rs=tmod:forward(td)
	if not rs[1]:equal(rs[2]) then
		print("forward passed")
	end
	grad=tmod:backward(td,grad)
	if not grad[1]:equal(grad[2]) then
		print("backward passed")
	end
	tmod:updateParameters(1/8192)
	print(_.."\t"..tmod.bias[1])
end
