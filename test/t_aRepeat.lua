require "nn"
require "aRepeat"

tmod=nn.aRepeat(2,1,1)
td=torch.randn(4,2)
grad=torch.randn(4,4)
print(tmod:forward(td))
tg=tmod:backward(td,grad)
print(tg:equal(grad:narrow(2,1,2)+grad:narrow(2,3,2)))
