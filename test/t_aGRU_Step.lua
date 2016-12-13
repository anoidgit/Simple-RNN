print("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

require "nn"
require "rnn"
require "prepare_module"

bsize=4
isize=4
osize=4
nstep=4

tmod=nn.aStepGRU(isize,osize)
tmod:training()

td={}
grad={}
rs={}
for i=1,nstep do
	local _ci = torch.randn(bsize,isize)
	table.insert(td,_ci)
	table.insert(grad,torch.randn(bsize,osize))
	table.insert(rs,tmod:updateOutput(_ci))
end
print(rs[1]:equal(rs[2]))
gradi={}
for i=nstep,1,-1 do
	table.insert(gradi,tmod:backward(td[i],grad[i]))
end
print(gradi[1]:equal(gradi[2]))
print(tmod.outputs)
