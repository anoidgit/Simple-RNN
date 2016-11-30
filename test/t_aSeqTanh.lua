print("set default tensor type")
torch.setdefaulttensortype('torch.FloatTensor')

require "nn"
require "aSeqTanh"

td={}
for i=1,4 do
	table.insert(td,torch.randn(2,4))
end
tmod1=nn.Tanh()
tmod2=nn.aSeqTanh()
for _,v in ipairs(td) do
	rs1=tmod1:forward(v)
	rs2=tmod2:forward(v)
	if rs1:equal(rs2) then
		print("forward passed")
	else
		print(rs1)
		print(rs2)
	end
end
grado={}
for i=1,4 do
	table.insert(grado,torch.randn(2,4))
end
for _k,v in ipairs(td) do
	gradu=grado[_k]
	grad1=tmod1:backward(v,gradu)
	grad2=tmod2:backward(v,gradu)
	if grad1:equal(grad2) then
		print("backward passed")
	else
		print(grad1)
		print(grad2)
	end
end
