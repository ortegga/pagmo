#~/bin/python
f = open("testrun.log").read()
all = f.split("\n\n")

i = 1 #counter to name your output files
for items in all:
	name = str(i)
	while len(name) < 4:
		name = "0" + name
	name = "run_" + name + ".dat"
	open(name,"w").write(items)
	i = i + 1

