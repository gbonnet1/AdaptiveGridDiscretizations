import os
import sys
import json
"""
This file tests for an annoying Markdown typesetting issue: 
a mathematical expression of the form
$$
	a
	+ b
$$
will render correctly on some markdown viewers (including the jupyter notebook editor), 
and not on others (including nbviewer).

The solution is to replace + with {+} in the start of a line, 
if the next character is a blank space. Likewise for - and *.

---
Another issue was encountered in an equation of the form
$$
<n
$$

--- 
Another issue occurs when a line contains a single starting comment tag, 
and the previous line is not blank.

blabla.
<!---
my comment
--->

---
Another issue is that the command \\rm is not interpreted in the same manner depending
on the markdown front-end : it may apply to all the following characters, on only one. 
Prefer \\mathrm.
"""

def TestMath(filepath,show=False):
	with open(filepath, encoding='utf8') as data_file:
		data = json.load(data_file)
	def showcell(cell):
		if show: print("(Cell contents) : \n", *cell["source"])
	for cell in data["cells"]:
		if cell['cell_type']!='markdown': continue
		eqn = None
		prevLine="\n"
		for line in cell['source']:
			if line=="$$" or line=="$$\n":
				eqn = "" if eqn is None else None
				continue
			if eqn is not None:
				eqn = eqn+line
				l = line.lstrip()
				if line[0]=='<' or (l[0] in ['+','-','*'] and l[1]==' '):
					print(f"--- Markdown displaymath issue in file {filepath} : ---")
					print(eqn)
					showcell(cell)
			if line==("<!---\n") and prevLine!="\n":
				print(f"--- Markdown comment issue in file {filepath} : ---")
				print([prevLine],line)
				showcell(cell)
			if "\\rm " in line:
				print(f"--- Mardown math issue in file {filepath},", 
					"prefer \\mathrm{bla} to {\\rm bla} ---")
				print(line)
				showcell(cell)

			prevLine = line


if __name__ == '__main__':
#	TestToc("Notebooks_Algo","Dense")
#	TestTocs("Notebooks_Algo")
#	TestTocss()
	kwargs = {"show":False}
	for key in sys.argv[1:]:
		assert key[:2]=="--" and key[2:] in kwargs
		kwargs[key[2:]]=True

	for dirname in os.listdir():
		if not dirname.startswith("Notebooks_"): continue
		for filename in os.listdir(dirname):
			if not filename.endswith(".ipynb"): continue
			TestMath(os.path.join(dirname,filename),**kwargs)
