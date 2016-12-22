#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

def ldd(df):
	rsd={}
	with open(df) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8")
				k,v=tmp.split("	")
				rsd[k]=v
	return rsd

def mapfile(srcf,rsf,mapd,unkid,addf,addl,padv,reverseOrder):
	maxlen=0
	maprs=[]
	with open(srcf) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8")
				tmp=tmp.split(" ")
				rs=[mapd.get(tmpu,unkid) for tmpu in tmp]
				maprs.append(rs)
				clen=len(rs)
				if clen>maxlen:
					maxlen=clen
	for mapu in maprs:
		clen=len(mapu)
		mapu.insert(0,addf)
		mapu.append(addl)
		if clen<maxlen:
			ext=[padv for i in xrange(maxlen-clen)]
			mapu.extend(ext)
		if reverseOrder:
			mapu.reverse()
	with open(rsf,"w") as fwrt:
		tmp="\n".join([" ".join(lind) for lind in maprs])
		fwrt.write(tmp.encode("utf-8"))

def mapfl(srcfl,rsfl,mapf):
	isEncoder=True
	mapd=ldd(mapf)
	addf=mapd["SOS"]
	addl=mapd["EOS"]
	unkid=mapd["UNK"]
	crs=0
	for srcf in srcfl:
		mapfile(srcf,rsfl[crs],mapd,unkid,addf,addl,"0",isEncoder)
		crs+=1

if __name__=="__main__":
	fhead="train"
	typ=".src"
	srcp="rs/"
	rsp="mapd/"
	nfile=43855
	mapfl([srcp+fhead+str(i+1)+typ for i in xrange(nfile)],[rsp+fhead+str(i+1)+typ for i in xrange(nfile)],"map"+typ)
