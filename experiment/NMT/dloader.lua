require "ldapi"

wvec=loadObject('datasrc/wvec.asc')
sizvec=wvec:size(2)

--colid={}
--colidx={}
mword,mwordt=ldall('datasrc/thd/train','i.asc','t.asc',nsam)--loadnt('datasrc/thd/train','i.asc','t.asc',tld,nsam)
devin,devt=ldall('datasrc/thd/eval','i.asc','t.asc',ndev)
unigrams=loadObject("datasrc/unigrams.asc")
nclass=unigrams:size(1)
eosid=nclass-1

eaddtrain=nsam*ieps
