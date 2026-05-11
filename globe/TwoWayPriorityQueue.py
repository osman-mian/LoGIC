#Thanks to: https://stackoverflow.com/a/19431276
import collections
#from blist import sorteddict
from sortedcontainers import SortedDict 

class TwoWayPriorityQueue(collections.abc.MutableMapping):

	def __init__(self, data):
		self.dict_ = {}
		#self.sorted_ = sorteddict()
		self.sorted_  = SortedDict()
		self.update(data)

	def __getitem__(self, key):
		return self.dict_[key]

	def __setitem__(self, key, value):
		# remove old value from sorted dictionary
		if key in self.dict_:
			self.__delitem__(key)
		# update structure with new value
		self.dict_[key] = value
		try:
			keys = self.sorted_[value]
		except KeyError:
			self.sorted_[value] = set([key])
		else:
			keys.add(key)	#to handle having more than 1 keys with similar values
		

	def __delitem__(self, key):
		value = self.dict_.pop(key)
		keys = self.sorted_[value]
		keys.remove(key)
		if not keys:
			del self.sorted_[value]

	def __iter__(self):
		try:		
			for value, keys in self.sorted_.items():
				for key in keys:
					yield key
		except Exception:
			return;
	
	def __len__(self):
		return len(self.dict_)

	def next_best(self):
		for value, keys in self.sorted_.items():
			for key in keys:
				return key;

	def removeEntry(self,key):
		value = self.dict_.pop(key)
		keys = self.sorted_[value]
		keys.remove(key)
		if not keys:
			del self.sorted_[value]
				
	def hasMore(self):
		return len(self.dict_)>0;

	def display(self):
		for kl in self.items():
			print(kl)



'''
d={};
d[(1,2,1)]=1
d[(1,2,2)]=2
d[(2,3,1)]=3
d[(2,3,2)]=4
x = TwoWayPriorityQueue(d)
print("-----------------")
print (list(x.items()))
x[(1,2,1)] = 10
print("-----------------")
print (list(x.items()))
x[(2,3,2)] = -55
print("-----------------")
print (list(x.items()))
print(len(x.items()));
print("-----------------")
x[(2,3,100)] = -155
print (list(x.items()))
print(len(x.items()));
print("-----------------")
'''
