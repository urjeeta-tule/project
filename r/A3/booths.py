from flask import Flask, render_template, request
app = Flask(__name__, template_folder="templates")

def getbinArray(num, bits=8):
	arr = []
	if num<0:
		twoComplement = 2**bits - abs(num)
		twoComplement = format(twoComplement, 'b')
		for bit in twoComplement:
			arr.append(int(bit))
		return arr
	else:
		numBin = format(num, 'b').zfill(bits)
		for bit in numBin:
			arr.append(int(bit))
		return arr

def rightshift(num):
	num.pop()
	num.insert(0, num[0])
	return num

def Add(num, Acc):
	carry = 0
	Answer = []
	for i in range(len(num)-1, -1, -1):
		check = num[i]+Acc[i]+carry
		carry = 0
		if check == 0 or check == 1:
			Answer.insert(0,check)
		elif check == 2:
			Answer.insert(0,0)
			carry = 1
		else:
			Answer.insert(0,1)
			carry = 1
	return Answer

def binTodec(num):
	answer = 0
	for i in range(len(num)):
		answer = answer + num[i]*(2**(len(num)-1-i))

	if(num[0]):
		answer = answer - 2**16

	return answer

def booths(num1, num2):
	num1Bin = getbinArray(num1)
	num2Bin = getbinArray(num2)
	num2BinNeg = getbinArray(-num2)
	A = [0 for _ in range(len(num1Bin))]
	Q = num1Bin
	Qzero = 0
	Answer = A+Q
	for i in range(len(A)):
		if Answer[-1] == 0 and Qzero == 1:
			A = Add(num2Bin, A)
			Answer = A+Q
		elif Answer[-1] == 1 and Qzero == 0:
			A = Add(num2BinNeg, A)
			Answer = A+Q
		Qzero = Answer[-1]
		Answer = rightshift(Answer)
		Q = Answer[8:]
		A = Answer[:8] 
	
	answerinDecimal = binTodec(Answer)
	return answerinDecimal

@app.route('/')
def index():
	return "This is Booths Multiplication<br>Click <a href='/booths'>here</a> to use it"

@app.route('/booths', methods=['GET', 'POST'])
def multiplication():
	if request.method=="GET":
		return render_template("booths.html")
	else:
		num1 = int(request.form['num1'])
		num2 = int(request.form['num2'])
		answer = booths(num1, num2)
		return "Multiplication of "+str(num1)+" and "+str(num2)+" is <b>"+str(answer)+"</b>"