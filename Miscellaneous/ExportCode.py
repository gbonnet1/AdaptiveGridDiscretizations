import nbformat 
import json
import sys
import os

from TestCode import ListNotebooks

result_path = "ExportedCode"

def ExportCode(inFName,outFName,update=False,show=False,root="../.."):
	with open(inFName, encoding='utf8') as data_file:
		data = json.load(data_file)
	output = [
		f"# Code automatically exported from notebook {inFName}\n",
		"# Do not modify\n",
		f'import sys; sys.path.append("{root}") # Path to import agd\n\n'
	]
	nAlgo = 0
	for c in data['cells']:
		if 'tags' in c['metadata'] and 'ExportCode' in c['metadata']['tags']:
			output.extend(c['source'])
			output.append('\n\n')
			nAlgo+=1
	output = ''.join(output)
	if nAlgo==0: return
	try:
		with open(outFName,'r',encoding='utf8') as output_file:
			output_previous = output_file.read()
	except FileNotFoundError:
		output_previous=""
	if output_previous==output: return
	print(f"Exported code changes for file {inFName}")
	if show:
		print("--- New code ---\n", output, 
			  "--- Old code ---\n", output_previous)
	if update:
		print("Exporting ", nAlgo, " code cells from notebook ", inFName, " in file ", outFName)
		with open(outFName,'w+', encoding='utf8') as output_file:
			output_file.write(output)

if __name__ == '__main__':

	kwargs = {key[2:]:True for key in sys.argv[1:] if key[:2]=='--' and '=' not in key}
	kwargs.update([key[2:].split('=') for key in sys.argv[1:] if key[:2]=='--' and '=' in key])
	notebook_filenames = [key for key in sys.argv[1:] if key[:2]!='--']
	if len(notebook_filenames)==0: notebook_filenames = ListNotebooks()

	for name in notebook_filenames:
		subdir,fname = os.path.split(name)
		ExportCode(name+'.ipynb',os.path.join(subdir,result_path,fname)+'.py',**kwargs)


