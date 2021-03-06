print("set default tensor type")
torch.setdefaulttensortype('torch.FloatTensor')

isize=4
osize=isize
seqlen=4
bsize=4

function getstd()
return nn.Sequential()
		:add(nn.ConcatTable()
			:add(nn.Sequential()
				:add(nn.NarrowTable(1,2))
				:add(nn.JoinTable(1, 1, true))
				:add(tmodCoreL:clone("weight","bias")))
			:add(nn.Sequential()
				:add(nn.SelectTable(-1))
				:add(tmodCoreM:clone("weight","bias"))))
		:add(nn.CAddTable())
		:add(nn.Sigmoid())
end

function comtable(tb1,tb2)
local rs=true
for _,v in ipairs(tb1) do
	if not v:equal(tb2[_]) then
		rs=false
		break
	end
end
return rs
end

require "nn"
require "aSeqSigmoid"

tmodCoreL=nn.Linear(isize+osize, osize)
tmodCoreM=nn.CMul(osize)
tmod1=nn.Sequential()
		:add(nn.ConcatTable()
			:add(nn.Sequential()
				:add(nn.NarrowTable(1,2))
				:add(nn.JoinTable(1, 1, true))
				:add(tmodCoreL))
			:add(nn.Sequential()
				:add(nn.SelectTable(-1))
				:add(tmodCoreM)))
		:add(nn.CAddTable())
		:add(nn.aSeqSigmoid(true))
tmod1:training()
td={}
grad={}
for k=1,seqlen do
	table.insert(td,{torch.randn(bsize,isize),torch.randn(bsize,osize),torch.randn(bsize,osize)})
	table.insert(grad,torch.randn(bsize,osize))
end
tmod2=getstd()
for i=1,seqlen do
	print(tmod1:forward(td[i]):equal(tmod2:forward(td[i])))
end
for i=seqlen,1,-1 do
	tmod2=getstd()
	tmod2:forward(td[i])
	print(comtable(tmod2:backward(td[i],grad[i]),tmod1:backward(td[i],grad[i])))
end
