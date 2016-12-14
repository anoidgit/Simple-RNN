#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

def lowerfile(srcf,rsf):
	with open(rsf,"w") as fwrt:
		with open(srcf) as frd:
			for line in frd:
				tmp=line.strip()
				if tmp:
					tmp=tmp.decode("utf-8")
					tmp=tmp.lower()
					tmp+="\n"
					fwrt.write(tmp.encode("utf-8"))

if __name__=="__main__":
	lowerfile("peval.targ","eval.targ")
