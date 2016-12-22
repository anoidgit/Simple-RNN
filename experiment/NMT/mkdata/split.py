#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

import math

def splitf(fsrcs,fsrct,freq,rsp):
	cursave=1
	minib=16
	maxdl=16
	maxbatch=64
	maxind=50*50
	maxind=float(maxind)
	maxlen=96
	with open(freq) as frd:
		with open(fsrcs) as frds:
			with open(fsrct) as frdt:
				cache=[]
				for line in frd:
					tmp=line.strip()
					if tmp:
						tmp=tmp.decode("utf-8")
						lgth,freq=tmp.split("	")
						lgth=int(lgth)
						freq=int(freq)
						if lgth>maxlen:
							for i in xrange(freq):
								frds.readline()
								frdt.readline()
						else:
							mab=min(maxbatch,math.ceil(maxind/lgth))# max batchsize
							ib=int(math.ceil(mab/2.0))# half batchsize
							mab=int(mab)# to int
							if freq>minib:# if current freq larger than minimal batchsize
								if cache:# if there are things in cache, flush it
									# count how many lines to copy
									nstore=0
									for cu in cache:
										nstore+=cu[-1]
									#copy src
									with open(rsp+str(cursave)+".src","w") as fwrt:
										for i in xrange(nstore):
											lind=frds.readline()
											lind=lind.strip()
											lind=lind.decode("utf-8")
											lind+="\n"
											fwrt.write(lind.encode("utf-8"))
									#copy target
									with open(rsp+str(cursave)+".targ","w") as fwrt:
										for i in xrange(nstore):
											lind=frdt.readline()
											lind=lind.strip()
											lind=lind.decode("utf-8")
											lind+="\n"
											fwrt.write(lind.encode("utf-8"))
									cursave+=1
									cache=[]
								if freq>mab:# if current freq larger then max batchsize
									while freq>mab:# copy until current freq less than max batchsize
										# copy src, half of max batchsize each time
										with open(rsp+str(cursave)+".src","w") as fwrt:
											for i in xrange(ib):
												lind=frds.readline()
												lind=lind.strip()
												lind=lind.decode("utf-8")
												lind+="\n"
												fwrt.write(lind.encode("utf-8"))
										with open(rsp+str(cursave)+".targ","w") as fwrt:
											for i in xrange(ib):
												lind=frdt.readline()
												lind=lind.strip()
												lind=lind.decode("utf-8")
												lind+="\n"
												fwrt.write(lind.encode("utf-8"))
										cursave+=1
										freq-=ib
								# write current length data
								# write src
								with open(rsp+str(cursave)+".src","w") as fwrt:
									for i in xrange(freq):
										lind=frds.readline()
										lind=lind.strip()
										lind=lind.decode("utf-8")
										lind+="\n"
										fwrt.write(lind.encode("utf-8"))
								# write target
								with open(rsp+str(cursave)+".targ","w") as fwrt:
									for i in xrange(freq):
										lind=frdt.readline()
										lind=lind.strip()
										lind=lind.decode("utf-8")
										lind+="\n"
										fwrt.write(lind.encode("utf-8"))
								cursave+=1
							else:# if current freq less than mini batch
								if cache:# if there are cache
									if cache[0][0]-lgth<maxdl:# if length are padding acceptable
										cache.append([lgth,freq])# add to cache
										nstore=0
										for cu in cache:
											nstore+=cu[-1]
										if nstore>minib:# if cache larger then minimal batchsize, store it
											# copy src
											with open(rsp+str(cursave)+".src","w") as fwrt:
												for i in xrange(nstore):
													lind=frds.readline()
													lind=lind.strip()
													lind=lind.decode("utf-8")
													lind+="\n"
													fwrt.write(lind.encode("utf-8"))
											# copy target
											with open(rsp+str(cursave)+".targ","w") as fwrt:
												for i in xrange(nstore):
													lind=frdt.readline()
													lind=lind.strip()
													lind=lind.decode("utf-8")
													lind+="\n"
													fwrt.write(lind.encode("utf-8"))
											cursave+=1
											cache=[]
									else:# the length are not acceptable, flush the cache
										nstore=0
										for cu in cache:
											nstore+=cu[-1]
										# copy src
										with open(rsp+str(cursave)+".src","w") as fwrt:
											for i in xrange(nstore):
												lind=frds.readline()
												lind=lind.strip()
												lind=lind.decode("utf-8")
												lind+="\n"
												fwrt.write(lind.encode("utf-8"))
										# copy target
										with open(rsp+str(cursave)+".targ","w") as fwrt:
											for i in xrange(nstore):
												lind=frdt.readline()
												lind=lind.strip()
												lind=lind.decode("utf-8")
												lind+="\n"
												fwrt.write(lind.encode("utf-8"))
										cursave+=1
										cache=[[lgth,freq]]
								else:
									cache=[[lgth,freq]]
				if cache:# after the whole frequence file readed, if there are cache, flush it
					nstore=0
					for cu in cache:
						nstore+=cu[-1]
					# copy src
					with open(rsp+str(cursave)+".src","w") as fwrt:
						for i in xrange(nstore):
							lind=frds.readline()
							lind=lind.strip()
							lind=lind.decode("utf-8")
							lind+="\n"
							fwrt.write(lind.encode("utf-8"))
					# copy target
					with open(rsp+str(cursave)+".targ","w") as fwrt:
						for i in xrange(nstore):
							lind=frdt.readline()
							lind=lind.strip()
							lind=lind.decode("utf-8")
							lind+="\n"
							fwrt.write(lind.encode("utf-8"))
					cache=[]
					cursave+=1
	print cursave-1

def splitfl(srcsfl,srctfl,frqfl,rsflp):
	for i in xrange(len(srcsfl)):
		splitf(srcsfl[i],srctfl[i],frqfl[i],rsflp[i])

if __name__=="__main__":
	fd=["train"]
	splitfl([i+"s.src" for i in fd],[i+"s.targ" for i in fd],[i+"f.txt" for i in fd],["rs/"+i for i in fd])
