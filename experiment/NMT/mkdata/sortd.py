#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

def sortf(fsrcs,fsrct,frss,frst,freqf):
	l=[]
	storage={}
	with open(fsrcs) as frds:
		with open(fsrct) as frdt:
			for line in frds:
				tmps=line.strip()
				tmpt=frdt.readline()
				tmpt=tmpt.strip()
				if tmps:
					tmps=tmps.decode("utf-8")
					tmpt=tmpt.decode("utf-8")
					lgth=len(tmps.split(" "))+len(tmpt.split(" "))
					if lgth in storage:
						storage[lgth].append((tmps,tmpt))
					else:
						l.append(lgth)
						storage[lgth]=[(tmps,tmpt)]
	l.sort(reverse=True)
	with open(frss,"w") as fwrts:
		with open(frst,"w") as fwrtt:
			with open(freqf,"w") as fwrtf:
				for lu in l:
					lgth=len(storage[lu])
					cwrtl=zip(*storage[lu])
					tmp="\n".join(cwrtl[0])+"\n"
					fwrts.write(tmp.encode("utf-8"))
					tmp="\n".join(cwrtl[-1])+"\n"
					fwrtt.write(tmp.encode("utf-8"))
					tmp=str(lu)+"	"+str(lgth)+"\n"
					fwrtf.write(tmp.encode("utf-8"))

def sortfl(srcsfl,srctfl,rssfl,rstfl,frqfl):
	for i in xrange(len(srcsfl)):
		sortf(srcsfl[i],srctfl[i],rssfl[i],rstfl[i],frqfl[i])

if __name__=="__main__":
	fd=["train"]
	sortfl([i+".src" for i in fd],[i+".targ" for i in fd],[i+"s.src" for i in fd],[i+"s.targ" for i in fd],[i+"f.txt" for i in fd])
