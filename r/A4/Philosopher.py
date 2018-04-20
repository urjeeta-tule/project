import time, threading, pymongo, pprint

class Philosopher(threading.Thread):
	user = "username"
	password = "password"
	ip = "ipaddress"
	url = "mongodb://"+user+":"+password+"@"+ip
	connection = pymongo.MongoClient(url)

	def __init__(self, index, name, l1, l2):
		threading.Thread.__init__(self)
		self.id = index
		self.name = name
		self.fork1 = l1
		self.fork2 = l2
		self.running = True

	def sendto(self):
		# push data to mongo db
		print "push data to mongo db"
		col = self.connection.testdb.dininew
		col.insert_one({"time":time.ctime(), "philosopher": self.id})

	def eating(self):
		#philosopher eating
		op = "Philosopher", self.id, "is eating"
		print op
		self.sendto()
		time.sleep(5)
		op = "Philosopher", self.id, "is done eating"
		print op

	def grab(self):
		f1, f2 = self.fork1, self.fork2
		while self.running:
			f1.acquire(True)

			if f2.acquire(False):
				break

			f1.release()
			f1,f2 = f2, f1

		else:
			#end the thread execution
			return

		#got both threads now eating
		self.eating()
		f2.release()
		f1.release()

	def run(self):
		while self.running:
			self.grab()

def main():
	members = []
	forks = []
	names = []

	for i in range(5):
		forks.append(threading.Lock())
		names.append("Philosopher"+str(i))

	for i in range(5):
		members.append(Philosopher(i, names[i], forks[i%5], forks[(i+1)%5]))

	for i in range(5):
		members[i].start()

	time.sleep(30)

	for i in range(5):
		members[i].running = False

	for i in range(5):
		members[i].join()

	print "Worked it all up"

	print "*****************-----*****************"
	print "printing data from mongodb"

	user = "username"
	password = "password"
	ip = "ipaddress"
	url = "mongodb://"+user+":"+password+"@"+ip
	connection = pymongo.MongoClient(url)

	collection = connection.testdb.dininew
	for document in collection.find():
		pprint.pprint(document)


main()