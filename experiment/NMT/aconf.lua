--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts configure for adam

	Version 0.0.1

]]

starterate = math.huge--warning:only used as init erate, not asigned to criterion

runid = "161217"

ieps = 1
warmcycle = 1
expdecaycycle = 4
gtraincycle = 64

nsam = 43855
ndev = 75

nreport = 20

--[[prld = 0.75
nfresh = 64
tld = 128]]

modlr = 1/8192/64

csave = 3

lrdecaycycle = 32

knegsample = 64

cycs = false--warning:this option need a lot of memory
savecycle = 32

device = 3
