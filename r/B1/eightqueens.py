import json
import unittest

class Test8Queens(unittest.TestCase):
	def test1(self):
		self.assertEqual(getInput("input1"),True)
	def test2(self):
		#this case fails cause bad configuration
		self.assertEqual(getInput("input2"),True)
	def test3(self):
		self.assertEqual(getInput("bro"),False)
		



def place(board, row, column):
	for c in range(column):
		if board[c]==row:
			return False
		elif abs(board[c]-row) == abs(c-column):
			return False
	return True


def Queens(board, column):
	#do stuff
	for row in range(8):
		if place(board, row, column):
			board[column]=row
			if column==7:
				return True
			else:
				if(Queens(board, column+1)):
					return True
				else:
					#backtracking
					board[column]= -1 
	if row==8:
		return False

def HailHydra(board):
	print("------------------****------------------")
	data = [['_']*8 for _ in xrange(8)]
	
	for i in xrange(8):
		data[i][board[i]]='Q'
	for i in xrange(8):
		print data[i]

def getInput(filename):
	try:
		f = open(filename, "r")
		position = json.load(f)
		position = position['start']
		board = [-1]*8
		if(position>=0 and position<8):
			board[0]=position

			if(Queens(board, 1)):
				HailHydra(board)
				return True
			else:
				print "Cant have board with initial queen at:", position
				return False
		else:
			print "Invalid input"
			return False
	except Exception as e:
		print "Exception:",e
		return False

getInput("input")

unittest.main()