st = ""
finalst = ""

with open("esfj1.txt","r") as afile:
	for line in afile:
		if line.strip():
			st = st + line.strip().replace("\t"," ") + " "

l = st.split("CHAPTER")
for j in range(len(l)):
	finalst = finalst + "ESFJ" + "\t" + l[j] + "\n"


with open("estj4.txt","r") as afile:
	for line in afile:
		if line.strip():
			st = st + line.strip().replace("\t"," ") + " "

l = st.split("CHAPTER")
for j in range(len(l)):
	finalst = finalst + "ESTJ" + "\t" + l[j] + "\n"


with open("isfj2.txt","r") as afile:
	for line in afile:
		if line.strip():
			st = st + line.strip().replace("\t"," ") + " "

l = st.split("Chapter")
for j in range(len(l)):
	finalst = finalst + "ISFJ" + "\t" + l[j] + "\n"


with open("isfp3.txt","r") as afile:
	for line in afile:
		if line.strip():
			st = st + line.strip().replace("\t"," ") + " "

l = st.split("CHAPTER")
for j in range(45):
	finalst = finalst + "ISFP" + "\t" + l[j] + "\n"


with open("istj1.txt","r") as afile:
	for line in afile:
		if line.strip():
			st = st + line.strip().replace("\t"," ") + " "

l = st.split("\n\nCHAPTER")
for j in range(len(l)):
	finalst = finalst + "ISTJ" + "\t" + l[j] + "\n"
	

print finalst
