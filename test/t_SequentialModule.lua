print("set default tensor type")
torch.setdefaulttensortype('torch.FloatTensor')

seql = 4
isize1 = 4
isize2 = 4
bsize = 4
osize = 4

require "nn"
require "aSeqJoinTable"
require "aSeqTanh"

tmodcore = nn.Linear(isize1 + isize2, osize)

tmod1 = nn.Sequential()
	:add(nn.aSeqJoinTable(1, 1, true))
	:add(tmodcore)
	:add(nn.aSeqTanh(true))
tmod1:training()
id = {}
grad = {}
for _ = 1, seql do
	local _cd = {torch.randn(bsize, isize1), torch.randn(bsize, isize2)}
	table.insert(id, _cd)
	table.insert(grad, torch.randn(bsize, osize))
end

for _ = 1, seql do
	local rs1 = tmod1:forward(id[_])
	tmod2 = nn.Sequential()
		:add(nn.JoinTable(1, 1))
		:add(tmodcore:clone("weight","bias"))
		:add(nn.Tanh())
	local rs2 = tmod2:forward(id[_])
	if rs1:equal(rs2) then
		print(_..":forward passed")
	else
		print(rs1)
		print(rs2)
	end
end
for _ = seql, 1, -1 do
	local rs1 = tmod1:backward(id[_], grad[_])
	tmod2 = nn.Sequential()
		:add(nn.JoinTable(1, 1))
		:add(tmodcore:clone("weight","bias"))
		:add(nn.Tanh())
	tmod2:forward(id[_])
	local rs2 = tmod2:backward(id[_], grad[_])
	for k,v in ipairs(rs1) do
		v2 = rs2[k]
		if v:equal(v2) then
			print(_..":backward passed")
		else
			print(_)
			print(v)
			print(v2)
		end
	end
end
