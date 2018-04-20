import threading
import xml.etree.ElementTree as et
import unittest

class TestCases(unittest.TestCase):
	def test1(self):
		self.assertEqual(takeInput("input"),True)
	def test2(self):
		self.assertEqual(takeInput("invalidfile"),False)
	def test3(self):
		self.assertEqual(takeInput("invalidfile2"), True)

class Sort:
	arr = []
	def __init__(self, inputarr):
		self.arr = inputarr

	def partition(self, start, end):
		pivot = self.arr[start]
		i=start
		for j in range(start+1, end+1):
			if pivot>self.arr[j]:
				i+=1
				self.arr[i], self.arr[j] = self.arr[j], self.arr[i]
		self.arr[i], self.arr[start] = pivot, self.arr[i]
		return i


	def quick(self, p, q):
		if p<q:
			r = self.partition(p,q)
			t1 = threading.Thread(target=self.quick, args=(p,r-1))
			t2 = threading.Thread(target=self.quick, args=(r+1,q))
			t1.start()
			t2.start()
			t1.join()
			print t1.getName()
			t2.join()
			print t2.getName()
		return self.arr


def takeInput(filename):
	try:
		f = open(filename, "r")
		tree = et.parse(f)
		root = tree.getroot()
		data = map(int, root.text.split(" "))
		sortobj = Sort(data)
		sortobj.quick(0,len(data)-1)
		print "Sorted Array", sortobj.arr
		return True
	except Exception as e:
		print "Exception occured", e
		return False

takeInput("input")
unittest.main()



