--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This was a training scripts to train a NMT model

	Version 0.0.1

]]

print("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

function feval()
	return _inner_err, _inner_gradParams
end

function gradUpdate(mlpin, x, y, criterionin, lr, optm)

	_inner_gradParams:zero()

	local pred=mlpin:forward(x)
	_inner_err=criterionin:forward(pred, y)
	sumErr=sumErr+_inner_err
	local gradCriterion=criterionin:backward(pred, y)
	pred=nil
	mlpin:backward(x, gradCriterion)

	optm(feval, _inner_params, {learningRate = lr})

	--mlpin:maxParamNorm(2)

end

function evaDev(mlpin, x, y, criterionin)
	mlpin:evaluate()
	local serr=0
	for curpot,v in ipairs(x) do
		serr=serr+criterionin:forward(mlpin:forward(v), y[curpot])
	end
	mlpin:training()
	return serr/ndev
end

--[[function inirand(cyc)
	cyc=cyc or 8
	for i=1,cyc do
		local sdata=math.random(nsam)
	end
end]]

function saveObject(fname,objWrt)
	if torch.isTensor(objWrt) then
		torch.save(fname,objWrt)
	else
		torch.save(fname,nn.Serial(objWrt):mediumSerial())
	end
	--[[local file=torch.DiskFile(fname,'w')
	file:writeObject(tmpod)
	file:close()]]
end

print("pre load package")
require "dpnn"
require "dp"

print("load settings")
require"conf"

print("load data")
require "dloader"

sumErr=0
crithis={}
cridev={}
erate=0
edevrate=0
storemini=1
storedevmini=1
minerrate=starterate
mindeverrate=minerrate

if cycs then
	bnnmod=nil
	bdevnnmod=nil
	bupdnm=false
	bupdevnm=false
end

function train()

	print("prepare environment")
	require "paths"
	local savedir="modrs/"..runid.."/"
	paths.mkdir(savedir)

	print("load optim")

	require "getoptim"
	local optmethod=getoptim()

	print("design neural networks and criterion")

	require "cunn"
	require "cunnx"

	require "designn"
	local nnmod=getnn()

	print(nnmod)
	nnmod:training()

	local critmod=getcrit()

	nnmod:cuda()
	critmod:cuda()

	_inner_params, _inner_gradParams=nnmod:getParameters()

	print("init train")
	local epochs=1
	local lr=modlr
	--inirand()

	print("turn off embeddings update")
	dupvec(nnmod)
	print("turn off embeddings norm")
	dnormvec(nnmod)
	
	mindeverrate=evaDev(nnmod,devin,devt,critmod)
	print("Init model Dev:"..mindeverrate)

	collectgarbage()

	print("start pre train")
	for tmpi=1,warmcycle do
		for tmpj=1,ieps do
			for curpot,v in ipairs(mword) do
				gradUpdate(nnmod,v,mwordt[curpot],critmod,lr,optmethod)
			end
		end
		local erate=sumErr/eaddtrain
		if erate<minerrate then
			minerrate=erate
		end
		table.insert(crithis,erate)
		print("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
		sumErr=0
		epochs=epochs+1
	end
	
	print("save neural network trained")
	saveObject(savedir.."nnmod.asc",nnmod)
	
	--[[print("turn on embeddings update")
	upvec(nnmod)]]

	epochs=1
	icycle=1

	aminerr=1
	lrdecayepochs=1
	
	collectgarbage()

	while true do
		print("start innercycle:"..icycle)
		for innercycle=1,gtraincycle do
			for tmpi=1,ieps do
				for curpot,v in ipairs(mword) do
					gradUpdate(nnmod,v,mwordt[curpot],critmod,lr,optmethod)
				end
			end
			local erate=sumErr/eaddtrain
			table.insert(crithis,erate)
			local edevrate=evaDev(nnmod,devin,devt,critmod)
			table.insert(cridev,edevrate)
			print("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate..",Dev:"..edevrate)
			--print("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
			local modsavd=false
			if edevrate<mindeverrate then
				mindeverrate=edevrate
				if cycs then
					nnmod:float()
					bdevnnmod=nnmod:clone()
					nnmod:cuda()
					bupdevnm=true
				else
					saveObject(savedir.."devnnmod"..storedevmini..".asc",nnmod)
					storedevmini=storedevmini+1
					if storedevmini>csave then
						storedevmini=1
					end
				end
				modsavd=true
				print("new minimal dev error found, model saved")
			end
			if erate<minerrate then
				minerrate=erate
				aminerr=1
				if not modsavd then
					if cycs then
						nnmod:float()
						bnnmod=nnmod:clone()
						nnmod:cuda()
						bupdnm=true
					else
						saveObject(savedir.."nnmod"..storemini..".asc",nnmod)
						storemini=storemini+1
						if storemini>csave then
							storemini=1
						end
					end
					print("new minimal error found, model saved")
				end
			else
				if aminerr>=expdecaycycle then
					aminerr=0
					if lrdecayepochs>lrdecaycycle then
						modlr=lr
						lrdecayepochs=1
					end
					lrdecayepochs=lrdecayepochs+1
					lr=modlr/(lrdecayepochs)
				end
				aminerr=aminerr+1
			end
			sumErr=0
			if cycs and epochs%savecycle==0 then
				if bupdevnm then
					print("flush dev mod")
					torch.save(savedir.."devnnmod"..storedevmini..".asc",bdevnnmod)
					bdevnnmod=nil
					storedevmini=storedevmini+1
					if storedevmini>csave then
						storedevmini=1
					end
					bupdevnm=false
				end
				if bupdnm then
					print("flush mod")
					torch.save(savedir.."nnmod"..storemini..".asc",bnnmod)
					bnnmod=nil
					storemini=storemini+1
					if storemini>csave then
						storemini=1
					end
					bupdnm=false
				end
			end
			epochs=epochs+1
		end

		print("save neural network trained")
		saveObject(savedir.."nnmod.asc",nnmod)

		print("save criterion history trained")
		local critensor=torch.Tensor(crithis)
		saveObject(savedir.."crit.asc",critensor)
		local critdev=torch.Tensor(cridev)
		saveObject(savedir.."critdev.asc",critdev)

		--[[print("plot and save criterion")
		gnuplot.plot(critensor)
		gnuplot.figprint(savedir.."crit.png")
		gnuplot.figprint(savedir.."crit.eps")
		gnuplot.plotflush()
		gnuplot.plot(critdev)
		gnuplot.figprint(savedir.."critdev.png")
		gnuplot.figprint(savedir.."critdev.eps")
		gnuplot.plotflush()]]

		critensor=nil
		critdev=nil

		print("task finished!Minimal error rate:"..minerrate.."	"..mindeverrate)
		--print("task finished!Minimal error rate:"..minerrate)

		print("wait for test, neural network saved at nnmod*.asc")

		icycle=icycle+1

		print("collect garbage")
		collectgarbage()

	end
end

train()
