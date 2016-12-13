#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

rs={}
typ=".targ"
fhead="train"
for i in xrange(63477):
	with open("mapd\\"+fhead+str(i+1)+typ) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8")
				tmp=tmp.split(" ")
				for tmpu in tmp:
					cid=int(tmpu)
					rs[cid]=rs.get(cid,0)+1
with open("unigrams"+typ,"w") as fwrt:
	for i in xrange(len(rs)):
		tmp=str(rs[i+1])+"\n"# nce only need the distribution of target and its index start from 1
		fwrt.write(tmp.encode("utf-8"))
