torch.setdefaulttensortype('torch.FloatTensor')

function convec(fsrc,frs,lsize)
	local file=io.open(fsrc)
	local num=file:read("*n")
	local rs={}
	while num do
		table.insert(rs,num)
		num=file:read("*n")
	end
	file:close()
	ts=torch.Tensor(rs)
	ts:resize(#rs/lsize,lsize)
	--[[file=torch.DiskFile(frs,'w')
	file:writeObject(ts)
	file:close()]]
	torch.save(frs,ts)
end

function convTensor(fsrc,frs,uselong,append0,del0)
	local file=io.open(fsrc)
	local num=file:read("*n")
	local rs={}
	if del0 then
		num=file:read("*n")
	end
	if append0 then
		table.insert(rs,append0)
	end
	while num do
		table.insert(rs,num)
		num=file:read("*n")
	end
	file:close()
	if uselong then
		ts=torch.LongTensor(rs)
	else
		ts=torch.Tensor(rs)
	end
	torch.save(frs,ts)
end

function convfile(fsrc,frs,uselong)
	local file=io.open(fsrc)
	local lind=file:read("*n")
	local num=file:read("*n")
	local rs={}
	while num do
		local tmpt={}
		for i=1,lind do
			table.insert(tmpt,num)
			num=file:read("*n")
		end
		table.insert(rs,tmpt)
	end
	file:close()
	if uselong then
		ts=torch.LongTensor(rs)
	else
		ts=torch.Tensor(rs)
	end
	torch.save(frs,ts)
	--[[file=torch.DiskFile(frs,'w')
	file:writeObject(ts)
	file:close()]]
end

function gvec(nvec,vecsize,frs)
	--[[local file=torch.DiskFile(frs,"w")
	file:writeObject(torch.randn(nvec,vecsize))
	file:close()]]
	torch.save(frs,torch.randn(nvec,vecsize))
end

--convec("wvec.txt","wvec.asc",256)
gvec(52474,1024,"wvec.asc")

convTensor("unigrams.targ","unigrams.asc")

for nf=1,63477 do
	convfile("duse/train"..nf..".src","thd/train"..nf.."i.asc",true)
	convfile("duse/train"..nf..".targ","thd/train"..nf.."t.asc",true)
end

for nf=1,75 do
	convfile("duse/eval"..nf..".src","thd/eval"..nf.."i.asc",true)
	convfile("duse/eval"..nf..".targ","thd/eval"..nf.."t.asc",true)
end
