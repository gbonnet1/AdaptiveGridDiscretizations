"""
This file collects functional like methods used throughout the AD library.
"""

# --------- (Recursive) iteration --------

def from_generator(iterable_type):
	"""
	Returns the method for constructing an object from a generator.
	"""
	return getattr(iterable_type,'from_generator',iterable_type)


def rec_iter(x,iterables):
	"""
	Iterate recursively over x. 
	In the case of dictionnaries, if specified among the iterables, one iterates over values.
	"""
	if isinstance(x,iterables):
		if isinstance(x,dict): x=x.values()
		for y in x: 
			for z in rec_iter(y,iterables): yield z
	else: yield x

class pair(object):
	"""
	A two element iterable. 
	Introduced as an alternative of tuple, to avoid confusion in map_iterables
	"""
	def __init__(self,first,second):
		self.first=first
		self.second=second
	def __iter__(self):
		yield self.first
		yield self.second
	def __str__(self):
		return "pair("+str(self.first)+","+str(self.second)+")"
	def __repr__(self):
		return "pair("+repr(self.first)+","+repr(self.second)+")"


def map_iterables(f,a,iterables,split=False): 
	"""
	Apply f to variable 'a' exploring recursively certain iterables
	"""
	if isinstance(a,iterables):
		type_a = type(a)
		if issubclass(type(a),dict):
			result = type_a({key:map_iterables(f,val,iterables,split=split) for key,val in a.items()})
			if split: return type_a({key:a for key,(a,_) in a.items()}), type_a({key:a for key,(_,a) in a.items()})
			else: return result
		else: 
			ctor_a = from_generator(type_a)
			result = ctor_a(map_iterables(f,val,iterables,split=split) for val in a)
			if split: return ctor_a(a for a,_ in result), ctor_a(a for _,a in result)
			else: return result 
	return f(a)

def map_iterables2(f,a,b,iterables):
	"""
	Apply f to variable 'a' and 'b' zipped, exploring recursively certain iterables
	"""
	for type_iterable in iterables:
		if isinstance(a,type_iterable):
			if issubclass(type_iterable,dict):
				return type_iterable({key:map_iterables2(f,a[key],b[key],iterables) for key in a})
			else: 
				return from_generator(type_iterable)(map_iterables2(f,ai,bi,iterables) for ai,bi in zip(a,b))
	return f(a,b)

# -------- Decorators --------

def recurse(step,niter=1):
	def operator(rhs):
		nonlocal step,niter
		for i in range(niter):
			rhs=step(rhs)
		return rhs
	return operator