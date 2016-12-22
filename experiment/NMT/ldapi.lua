require "cutorch"
device = device or 1
cutorch.setDevice(device)
print("Model will be trained on device:"..device)

function loadObject(fname)
	--[[local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()]]
	return torch.load(fname)
end

function ldall(iprefix,ifafix,tfafix,ntotal)
	local id={}
	local tart={}
	for i=1,ntotal do
		table.insert(id,loadObject(iprefix..i..ifafix):cuda())
		table.insert(tart,loadObject(iprefix..i..tfafix):cuda())
	end
	return id,tart
end

function loadnt(iprefix,ifafix,tfafix,nfile,ntotal)
	local id={}
	local tart={}
	local curld
	for i=1,nfile do
		curld=math.random(ntotal)
		while colid[curld] do
			curld=math.random(ntotal)
		end
		colid[curld]=true
		table.insert(colidx,curld)
		table.insert(id,loadObject(iprefix..curld..ifafix):cuda())
		table.insert(tart,loadObject(iprefix..curld..tfafix):cuda())
	end
	return id,tart
end

function prod(modin,dtin)

	local rs={}
	for k,v in ipairs(dtin) do
		table.insert(rs,modin:updateOutput(v))
	end
	return rs

end

function rldc()
	for _tmpi=1,nfresh do
		table.remove(mword,1)
		table.remove(mwordt,1)
	end
	local apin,apt=loadnt('datasrc/thd/train','i.asc','t.asc',nfresh,nsam)
	for _tmpi=1,nfresh do
		table.insert(mword,table.remove(apin))
		table.insert(mwordt,table.remove(apt))
		table.remove(colid,table.remove(colidx,1))
	end
end

function rldt()
	rldc()
	collectgarbage()
end
