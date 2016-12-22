local Sequencer = nn.Sequencer

function Sequencer:backward(input, gradOutput, scale)
   local nStep
   if torch.isTensor(input) then
      assert(torch.isTensor(gradOutput), "expecting gradOutput Tensor since input is a Tensor")
      assert(gradOutput:size(1) == input:size(1), "gradOutput should have as many elements as input")
      nStep = input:size(1)
   else
      assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
      assert(#gradOutput == #input, "gradOutput should have as many elements as input")
      nStep = #input
   end

   -- back-propagate through time
   self.tablegradinput = {}
   for step=nStep,1,-1 do
      self.tablegradinput[step] = self.module:backward(input[step], gradOutput[step], scale)
   end

   if torch.isTensor(input) then
      self.gradInput = torch.isTensor(self.gradInput) and self.gradInput or self.tablegradinput[1].new()
      self.gradInput:resize(nStep, unpack(self.tablegradinput[1]:size():totable()))
      for step=1,nStep do
         self.gradInput[step]:copy(self.tablegradinput[step])
      end
   else
      self.gradInput = self.tablegradinput
   end

   return self.gradInput
end