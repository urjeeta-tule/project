import unittest

class TestBinSearch(unittest.TestCase):
	def test1(self):
		self.assertEqual(takeInput("input", 8), True)
	def test2(self):
		self.assertEqual(takeInput("ggwp", 8), False)

class BinarySearch:
	arr = []

	def __init__(self, a):
		self.arr = a

	def Partition(self, p, q):
		pivot = self.arr[p]
		i = p
		for j in range(p+1, q+1):
			if(self.arr[j]<pivot):
				i+=1
				self.arr[i], self.arr[j] = self.arr[j], self.arr[i]
		self.arr[i], self.arr[p] = self.arr[p], self.arr[i]
		return i

	def QuickSort(self, p, q):
		if p<q:
			r = self.Partition(p, q)
			self.QuickSort(p, r-1)
			self.QuickSort(r+1, q)
		else:
			return self.arr

	def Search(self, num, start, end):
		if(start<end):
			mid = start+(end-start)/2
			if(self.arr[mid]==num):
				return mid
			elif(self.arr[mid]>num):
				return self.Search(num, start, mid-1)
			else:
				return self.Search(num, mid+1, end)
		return -1



def takeInput(fileName, searchNum):
	try:
		file = open(fileName, "r")
		numbers = []
		for line in file:
			numbers.append(int(line))
		binSearch = BinarySearch(numbers)
		binSearch.QuickSort(0, len(numbers)-1)
		if binSearch.Search(searchNum, 0, len(numbers)):
			print("number found")
			return True
		else:
			return False
	except Exception as e:
		print "Exception123: "+ str(e)
	return False

def main():
	filename = raw_input("Enter filename> ")
	numSearch = input("Enter number to search> ")
	takeInput(filename, numSearch)

main()

unittest.main()