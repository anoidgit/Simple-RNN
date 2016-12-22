#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

nearwords=50000

freqd={}
ends=".targ"
nline=0
for i in xrange(32694):
	with open("rs/train"+str(i+1)+ends) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				nline+=1
				tmp=tmp.decode("utf-8")
				tmp=tmp.split(" ")
				for tmpu in tmp:
					freqd[tmpu]=freqd.get(tmpu,0)+1
freql=[]
freqw={}
with open("freq"+ends,"w") as fwrt:
	for k,v in freqd.iteritems():
		if not v in freqw:
			freql.append(v)
			freqw[v]=[k]
		else:
			freqw[v].append(k)
		tmp=k+"	"+str(v)+"\n"
		fwrt.write(tmp.encode("utf-8"))
freql.sort(reverse=True)
cwrt=1
writeall=True
unigrams={}
ncunk=True
with open("map"+ends,"w") as fwrt:
	for freq in freql:
		wl=freqw[freq]
		for word in wl:
			if ncunk:
				unigrams[cwrt]=freq
				tmp=word+"	"+str(cwrt)+"\n"
				fwrt.write(tmp.encode("utf-8"))
				cwrt+=1
			else:
				unigrams[unkid]+=freq
		if cwrt>nearwords:
			if ncunk:
				writeall=False
				for word in ["SOS","EOS"]:
					tmp=word+"	"+str(cwrt)+"\n"
					fwrt.write(tmp.encode("utf-8"))
					unigrams[cwrt]=nline
					cwrt+=1
				unkid=cwrt
				unigrams[unkid]=0
				ncunk=False
				print(str(unkid)+" words maped")
	if writeall:
		for word in ["SOS","EOS"]:
			tmp=word+"	"+str(cwrt)+"\n"
			fwrt.write(tmp.encode("utf-8"))
			unigrams[cwrt]=nline
			cwrt+=1
		unkid=cwrt
		unigrams[unkid]=0
		print(str(unkid)+" words maped")
	tmp="UNK	"+str(unkid)
	fwrt.write(tmp.encode("utf-8"))
with open("unigrams"+ends,"w") as fwrt:
	for i in xrange(len(unigrams)):
		tmp=str(unigrams[i+1])+"\n"
		fwrt.write(tmp)