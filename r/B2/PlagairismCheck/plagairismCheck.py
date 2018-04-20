from flask import Flask, render_template, request
from difflib import SequenceMatcher
app = Flask(__name__, template_folder='templates')

completeText = ""
file = open("text1.txt","r")
# completeText+=file.read()
text1 = []
for line in file:
	text1.append(line.strip())

file = open("text2.txt","r")
# completeText+=file.read()
text2 = []
for line in file:
	text2.append(line.strip())

file = open("text3.txt","r")
# completeText+=file.read()
text3 = []
for line in file:
	text3.append(line.strip())

@app.route('/')
def hello_world():
	header = 'Hello, World!<br>This is Plagairism Checker<br>'
	linktext = 'Click <a href="/check">here</a> to use it'
	return header+linktext

@app.route('/check',methods=['GET', 'POST'])
def check():
	if request.method == 'POST':
		response = ""
		text = request.form['PlagairismText']
		lines = text.split("\n")
		textlenght = len(lines)
		rendertext = ""
		for line in lines:
			rendertext = rendertext+'<br>'+line
		response = "Given Text is as follows<br>"+rendertext

		matches = 0

		textdatabase = [text1, text2, text3]
		for file in textdatabase:
			for line in lines:
				if line.strip() in file:
					matches+=1

		plagairismPercentage = (float(matches)/textlenght)*100
		# plagairismPercentage = SequenceMatcher(None, text, completeText).ratio()
		
		# if plagairismPercentage1>plagairismPercentage:
		# 	plagairismPercentage = plagairismPercentage1

		plagairismOutput = "The given text has <b>"+str(plagairismPercentage)+" %</b> plagairised text"
		response = response+'<br><br>'+plagairismOutput

		return response
	else:
		return render_template("plagairismForm.html")

