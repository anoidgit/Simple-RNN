--[[
	The GNU GENERAL PUBLIC LICENSE Version 3

	Copyright (c) 2016 Hongfei Xu

	This scripts configure for adam

	Version 0.0.1

]]

starterate = math.huge--warning:only used as init erate, not asigned to criterion

runid = "161208"

ieps = 2
warmcycle = 0
expdecaycycle = 2
gtraincycle = 64

modlr = 1/1024

csave = 3

nclass = unigrams:size(1)

lrdecaycycle = 4

cycs = false--warning:this option need a lot of memory
savecycle = 32
