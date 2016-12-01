print("set default tensor type")
torch.setdefaulttensortype('torch.FloatTensor')

require "nn"
require "aSigmoid"

td=torch.randn(2,4)
tmod1=nn.Sigmoid()
tmod2=nn.aSigmoid()
rs1=tmod1:forward(td)
rs2=tmod2:forward(td)
if rs1:equal(rs2) then
	print("forward passed")
else
	print(rs1)
	print(rs2)
end
grado=torch.randn(2,4)
grad1=tmod1:backward(td,grado)
grad2=tmod2:backward(td,grado)
if grad1:equal(grad2) then
	print("backward passed")
else
	print(grad1)
	print(grad2)
end
