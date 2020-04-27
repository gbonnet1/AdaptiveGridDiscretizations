import json
import os
import sys

from TestCode import ListNotebooks

"""
This file slightly modifies the agd notebooks and exports them, 
for use in google Colab environnement
"""

"""
links = {
	"Notebooks_Algo/ADBugs.ipynb",
	"Notebooks_Algo/Dense.ipynb",
	"Notebooks_Algo/FiniteDifferences.ipynb",
	"Notebooks_Algo/Reverse.ipynb",
	"Notebooks_Algo/Sparse.ipynb",
	"Notebooks_Algo/SternBrocot.ipynb",
	"Notebooks_Algo/SubsetRd.ipynb",
	"Notebooks_Algo/
"""

google_drive_link = {
	"Notebooks_GPU/Isotropic_Repro.ipynb":"https://drive.google.com/open?id=1YXqLXOkJXs1jfmNsp1IKwn3-mdpelKQx",
	"Notebooks_FMM/Isotropic.ipynb":"https://drive.google.com/open?id=blabla",
	"Notebooks_Algo/ADBugs.ipynb":"https://drive.google.com/open?id=1XeyaVEuGZpl-ebGDyQ9Dt6RB8NggItst",
	}

def Links():
	links = {}
	for key,value in google_drive_link.items():
		link_prefix = 'https://drive.google.com/open?id='
		assert value.startswith(link_prefix)
		drive_id = value[len(link_prefix):]
		colab_link = f"https://colab.research.google.com/notebook#fileId={drive_id}&offline=true&sandboxMode=true"
		subdir,filename = os.path.split(key)
		links.update({
			filename:colab_link,
			key:colab_link,
			"../"+key:colab_link,
			})
	return links

def ToColab(filename,output_dir):
	with open(filename+'.ipynb', encoding='utf8') as data_file:
		data=json.load(data_file)

	# Import the agd package from pip
	for cell in data['cells']:
		if (cell['cell_type']=='code' 
			and cell['source'][0].startswith('import sys; sys.path.insert(0,"..")')):
			cell['source'] = ['pip install agd']
			# Do not forget to turn on GPU mode in Google Colab (R) parameters if necessary
			break
	else: assert 'Summary' in filename

	links = Links()
	# Change the links to open in colab
	for cell in data['cells']:
		if cell['cell_type']=='markdown':
			for i,line in enumerate(cell['source']):
				if "](" in line:
					orig=line
					for key,value in links.items():
						line = line.replace("]("+key,"]("+value)
					if orig!=line:
						cell['source'][i]=line

	with open(os.path.join(output_dir,filename+'_Colab.ipynb'),'w') as f:
		json.dump(data,f,ensure_ascii=False)

def Main(output_dir):
	ToColab("Notebooks_Algo/ADBugs",output_dir)
#	for filename in ListNotebooks():
#		ToColab(filename,output_dir)

if __name__ == "__main__":
	for key in sys.argv[1:]:
		prefix = '--output_dir='
		if key.startswith(prefix):
			output_dir = key[len(prefix):]
			Main(output_dir)
			break
	else: print("Missing output_dir=...")



