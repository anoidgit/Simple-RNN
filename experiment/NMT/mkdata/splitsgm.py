
def parsedoc(strin):
	ind=strin.find("\"")
	tmp=strin[ind+1:]
	ind=strin.find("\"")
	tmp=tmp[:ind]
	return tmp

def parseg(strin):
	lind=strin.find(">")
	rind=strin.rfind("<")
	tmp=strin[lind+1:rind]
	return tmp.strip()

rsd={}
cache={}
dl=[]

with open("ref.sgm") as frd:
	for line in frd:
		tmp=line.strip()
		if tmp:
			tmp=tmp.decode("utf-8")
			if tmp.startswith("<DOC docid=\""):
				docid=parsedoc(tmp)
				if docid in cache:
					if not docid in rsd:
						rsd[docid]=[cache[docid]]
					else:
						rsd[docid].append(cache[docid])
				else:
					dl.append(docid)
				cache[docid]=[]
			else:
				if tmp.startswith("<seg id"):
					cache[docid].append(parseg(tmp))

for docid,v in cache.iteritems():
	if not docid in rsd:
		rsd[docid]=[cache[docid]]
	else:
		rsd[docid].append(cache[docid])
del cache

for i in xrange(4):
	with open("ref.plain"+str(i),"w") as fwrt:
		for doc in dl:
			tmp=rsd[doc].pop()
			tmp="\n".join(tmp)+"\n"
			fwrt.write(tmp.encode("utf-8"))
