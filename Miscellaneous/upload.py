import os
import sys
import shutil

import TestTocs
import TestMarkdown
import TestVersion
import ExportCode
import ExportColab
from TestCode import ListNotebooks

"""
This file will 
- test for basic errors in Tocs and markdown,
- update the version
- upload to conda and pip
- cleanup (rm -r mydir)
- create the google drive variants and move them
"""

showcase_dir = "../AdaptiveGridDiscretizations_showcase"

def Main_GPU():
	for filepath in ListNotebooks():
		if not filepath.startswith('Notebooks_GPU'): continue
		dirname,filename = os.path.split(filepath)
		shutil.copyfile(os.path.join(dirname,'test_results',filename+'_out.ipynb'),
			os.path.join(showcase_dir,filepath+'.ipynb'))

def Main(new_version=None,gpu=False):
	if gpu: return Main_GPU()
	# Routine checks. Code must be checked independently
	TestTocs.Main(check_raise=True)
	TestMarkdown.Main(check_raise=True)

	# Routine exports
	colab_dir = "/Users/mirebeau/Google Drive/AdaptiveGridDiscretizations_Colab"
	ExportCode.Main(update=True)
	ExportColab.Main(output_dir=colab_dir)

	for filepath in ListNotebooks():
		if filepath=='Summary':
			shutil.copyfile('Summary.ipynb',os.path.join(showcase_dir,'Summary.ipynb'))
			continue
		elif filepath.startswith('Notebooks_GPU'):
			continue

		dirname,filename = os.path.split(filepath)
		shutil.copyfile(os.path.join(dirname,'test_results',filename+'_out.ipynb'),
			os.path.join(showcase_dir,filepath+'ipynb'))

	raise ValueError

	# New version number
	if 'new_version' in kwargs:
		new_version = kwargs['new_version']
	else:
		old_version = TestVersion.Main()
		old_version = old_version.split('.')
		old_version[-1]=str(int(old_version[-1])+1) # Update minor version
		new_version = '.'.join(old_version)
		print("new_version :",new_version)

	TestVersion.Main(new_version)

	
	# Build conda and pip versions
	os.system("conda build .")
	os.system("python conda.recipe/setup.py bdist_wheel")
	
	# Export
	filename = f"/Users/mirebeau/opt/miniconda3/envs/agd-hfm_dev/conda-bld/noarch/agd-{new_version}-py_0.tar.bz2"
	os.system(f"anaconda upload {filename} --user agd-lbr")
	os.system("python -m twine upload dist/*")

	# Cleanup
	os.system("rm -r dist")
	os.system("rm -r build")
	os.system("rm -r agd.egg-info")


if __name__=="__main__":
	kwargs = {}
	for keyval in sys.argv[1:]:
		assert keyval.startswith('--')
		if '=' in keyval:
			key,value = keyval[2:].split('=')
		else:
			key=keyval[2:]
			value = True
		kwargs[key]=value 

	Main(**kwargs)






	