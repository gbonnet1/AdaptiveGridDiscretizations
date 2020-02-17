import json
import sys
import os

"""
Produce an exercise notebook, from a standard notebook with some special 
tags in the cell metadata, and comments in the cell source. 

TODO : A comment only version might be preferable, since the use of tags 
is rather redundant and error prone.
"""

indexExo = 0
language = 'FR'

def SplitExo(c):
	text = []
	comment = []
	statementFR =[]
	statementEN=[]
	code = []
	current = text
	for line in c['source']:
		if line == '<!---ExoFR\n':
			assert current is text
			current = statementFR
			continue
		elif line == '<!---ExoEN\n':
			assert current is text
			current = statementEN
			continue
		elif line == '<!---ExoCode\n':
			assert current is text
			current = code
			continue
		elif line == '<!---\n':
			assert current is text
			current = comment
			continue
		elif line == '--->' or line =='--->\n':
			assert current is not text
			current = text
			continue
		current.append(line)
	assert current is text
	c['source'] = text
	result = [c]
	global language,indexExo
	if len(statementFR)!=0 and language=='FR':
		indexExo+=1
		result.append({
			'cell_type':'markdown',
			'source':["*Question "+str(indexExo)+"*\n","===\n"]+statementFR,
			'metadata':{}
			})
	if len(statementEN)!=0 and language=='EN':
		indexExo+=1
		result.append({
			'cell_type':'markdown',
			'source':["*Question "+str(indexExo)+"*\n","===\n"]+statementEN,
			'metadata':{}
			})
	if len(code)!=0:
		result.append({		
		'cell_type':'code',
		'source':code,
		'execution_count':None,
		'outputs':[],
		'metadata':{},
		})
	return result

def MakeExo(FileName,ExoName):
	with open(FileName, encoding='utf8') as data_file:
		data=json.load(data_file)
	newcells = []
	removeCell = False
	for c in data['cells']:
		if 'tags' in c['metadata']:
			tags = c['metadata']['tags']
			if 'ExoRemove' in tags or removeCell:
				removeCell=False
				continue
			elif 'ExoSplit' in tags or c['cell_type']=='markdown':
				if "<!---ExoCode\n" in c['source']:
					removeCell=True # Remove next cell
				for x in SplitExo(c):
					newcells.append(x)
				continue
		newcells.append(c)
	data['cells']=newcells

	with open(ExoName,'w') as f:
		json.dump(data,f,ensure_ascii=False)

if __name__ == '__main__':
	for name in sys.argv[1:]:
		dir,FileName = os.path.split(name)
		prefix,ext = os.path.splitext(FileName)
		ExoName = os.path.join(dir,"Exo",prefix+'_Exo.ipynb')
		MakeExo(FileName,ExoName)