#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

nearwords=200000

freqd={}
ends=".targ"
for i in xrange(40840):
	with open("rs/train"+str(i+1)+ends) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
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
with open("map"+ends,"w") as fwrt:
	for freq in freql:
		wl=freqw[freq]
		for word in wl:
			tmp=word+"	"+str(cwrt)+"\n"
			fwrt.write(tmp.encode("utf-8"))
			cwrt+=1
		if cwrt>nearwords:
			print(str(cwrt+2)+" words maped")
			writeall=False
			break
	for word in ["SOS","EOS","UNK"]:
			tmp=word+"	"+str(cwrt)+"\n"
			fwrt.write(tmp.encode("utf-8"))
			cwrt+=1
if writeall:
	print(str(cwrt-1)+" words maped")
