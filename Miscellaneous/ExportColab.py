import json
import os
import sys

from TestCode import ListNotebooks
from TestTocs import UpdateConfig

"""
This file slightly modifies the agd notebooks and exports them, 
for use in google Colab environnement
"""

google_drive_link = {
	"Notebooks_Algo/ADBugs.ipynb":"https://drive.google.com/open?id=1XeyaVEuGZpl-ebGDyQ9Dt6RB8NggItst",
	"Notebooks_Algo/Dense.ipynb":"https://drive.google.com/open?id=19pwvmrLbo8RaALUsyMqVpHU1GiUnGoU0",
	"Notebooks_Algo/FiniteDifferences.ipynb":"https://drive.google.com/open?id=1iIgL5x3CudMUOD0wPDdvzT1lT98be7Lz",
	"Notebooks_Algo/Reverse.ipynb":"https://drive.google.com/open?id=1ISFlxfNc3cOn23qGT6q-n7efinF38Pju",
	"Notebooks_Algo/Sparse.ipynb":"https://drive.google.com/open?id=1wBo5hQltiNJx4CuBqDrHX143CeAQhAsX",
	"Notebooks_Algo/SternBrocot.ipynb":"https://drive.google.com/open?id=1y35pPXTjzVXiKPA46vKwMV2PsF40G00d",
	"Notebooks_Algo/SubsetRd.ipynb":"https://drive.google.com/open?id=18sRTS61Z9PsIl6Uy-IFwsGLiVKsN6rlW",
	"Notebooks_Algo/Summary.ipynb":"https://drive.google.com/open?id=1_wj1rf6howKn0ZZT_32KqLQLJEwbn2yv",
	"Notebooks_Algo/TensorSelling.ipynb":"https://drive.google.com/open?id=1dMHXDYJoQBI_EtvQTcR15pbJaiP03S0p",
	"Notebooks_Algo/TensorVoronoi.ipynb":"https://drive.google.com/open?id=1-c8lSdOG5g94ZxnisBYRANBMe_Q1gX1k",
	"Notebooks_Algo/VoronoiVectors.ipynb":"https://drive.google.com/open?id=1YZHaNtmC3tw7g69tkJ1LRRwy-BrHUJF9",
	
	"Notebooks_Div/Elliptic.ipynb":"https://drive.google.com/open?id=1OXsyoDW0ifgmgmFhn_HPjLDY9OZcOdxW",
	"Notebooks_Div/EllipticAsymmetric.ipynb":"https://drive.google.com/open?id=1kWed3TD_L0snKlsXTTrbeiS_hgVlw9c-",
	"Notebooks_Div/Summary.ipynb":"https://drive.google.com/open?id=1kDQItj4cqfKXglkBAG_AIZ_X5Obf62Eg",
	"Notebooks_Div/Time1D_Div.ipynb":"https://drive.google.com/open?id=12fuUVP1BHfJhWOupAL4wLADSf9VYPyjM",
	"Notebooks_Div/VaradhanGeodesics.ipynb":"https://drive.google.com/open?id=1TJkrHpN78NXXPvH-0d6mbi80ug-pyCst",

	"Notebooks_FMM/AsymmetricQuadratic.ipynb":"https://drive.google.com/open?id=1iDcflF_548PcsaUKlJRsIWRv-vUjKpKM",
	"Notebooks_FMM/Curvature.ipynb":"https://drive.google.com/open?id=1gX9_1qsDcUkRktdkYKgd8c0q4hQt26_V",
	"Notebooks_FMM/Curvature3.ipynb":"https://drive.google.com/open?id=1AGF2jSzt518wH9AhZJQX_-QMHQd6-Nbg",
	"Notebooks_FMM/DeviationHorizontality.ipynb":"https://drive.google.com/open?id=1x309yEmht-G8sy9dxJN2LGeOE4INdnmW",
	"Notebooks_FMM/DubinsZermelo.ipynb":"https://drive.google.com/open?id=1buUB_AvIojeV1hSi3YS-lTyXoUbY-V_k",
	"Notebooks_FMM/FisherRao.ipynb":"https://drive.google.com/open?id=1XRXG0YSoZxU8Ka0AtSucsfdQYszFfyXO",
	"Notebooks_FMM/Geodesics.ipynb":"https://drive.google.com/open?id=1Y3c8mi0GQXQnbCanGIw8bkmYn6S_7fhU",
	"Notebooks_FMM/HighAccuracy.ipynb":"https://drive.google.com/open?id=1xDpCHu3ZImU3sO1GNYek1NqGMc9STNoi",
	"Notebooks_FMM/Illusion.ipynb":"https://drive.google.com/open?id=1WO0JxVUWjOGONCj9BtXINdhjRO5vCpb1",
	"Notebooks_FMM/Isotropic.ipynb":"https://drive.google.com/open?id=185m7jMuSm-sZsIV8rrrcjWaQTJu-JqHX",
	"Notebooks_FMM/MedialAxis.ipynb":"https://drive.google.com/open?id=1UtQNzy5j8oyrjM_zRxTL8mO-YONbPT8E",
	"Notebooks_FMM/Rander.ipynb":"https://drive.google.com/open?id=1gcLkfumeQATvx5kJlxdrXcLX7v2fRka2",
	"Notebooks_FMM/Riemannian.ipynb":"https://drive.google.com/open?id=1EIm2xgwydixphdLh0BDg_qnoMy13KFhY",
	"Notebooks_FMM/Seismic.ipynb":"https://drive.google.com/open?id=1ord6tSAJVofu73e8U27Wu_HyWtQr1RO-",
	"Notebooks_FMM/Sensitivity.ipynb":"https://drive.google.com/open?id=1v4iR88w_q4gQrsgUFpazmX5evFw7oHGt",
	"Notebooks_FMM/SensitivitySL.ipynb":"https://drive.google.com/open?id=1WNW4Aze7dudMkdZvbHmpx_oi_NBIjeUt",
	"Notebooks_FMM/SmartIO.ipynb":"https://drive.google.com/open?id=1K6NHIVGf_z78BrZowEJKERmu_4ZAvqaA",
	"Notebooks_FMM/Summary.ipynb":"https://drive.google.com/open?id=1iRwPOy_RqK_JRDlZcC_CiRxX5UbDzCBR",
	"Notebooks_FMM/Tubular.ipynb":"https://drive.google.com/open?id=1AMLRDrNQVXLnOdsracx4ZSwPR7bgi3J9",

	"Notebooks_NonDiv/EikonalEulerian.ipynb":"https://drive.google.com/open?id=14tyqvW56QD9EThM2tV00atBc61ZMV9oA",
	"Notebooks_NonDiv/LinearMonotoneSchemes2D.ipynb":"https://drive.google.com/open?id=1-DiCaMxXGbK0IUQeUd_UrWdjgB88f5Z_",
	"Notebooks_NonDiv/MongeAmpere.ipynb":"https://drive.google.com/open?id=12l4IB3b4q-GbLsBlDLACZeUTg8bnhYnT",
	"Notebooks_NonDiv/MonotoneSchemes1D.ipynb":"https://drive.google.com/open?id=1uuVuj49ziCgtHHujBJXf-d1QrwmYzAru",
	"Notebooks_NonDiv/NonlinearMonotoneFirst2D.ipynb":"https://drive.google.com/open?id=1WrLRUlJi-TdRHJSr8bnSkarBmmwrxBwy",
	"Notebooks_NonDiv/NonlinearMonotoneSecond2D.ipynb":"https://drive.google.com/open?id=1-D3TaNsONLK2FFBNz2VEC8OZqDJzGmsn",
	"Notebooks_NonDiv/OTBoundary1D.ipynb":"https://drive.google.com/open?id=187F61LigLuPjIheWSxYycJ9wh2-J_gOW",
	"Notebooks_NonDiv/ShapeFromShading.ipynb":"https://drive.google.com/open?id=1ENdnH9FmlhQGvef0TfCS7e3xiTMOMHXP",
	"Notebooks_NonDiv/Summary.ipynb":"https://drive.google.com/open?id=1Cy4aftpK3g769vI9y6oTJKC-y9EpTMr8",
	"Notebooks_NonDiv/Time1D_NonDiv.ipynb":"https://drive.google.com/open?id=17NF1LCE5HYp1lU5nvORQEQrN5grLKLT6",
	"Notebooks_GPU/Summary.ipynb":"https://drive.google.com/open?id=17ZU6QxzY9fludRpXmBCh3u9nEcv_n_ph",

	"Summary.ipynb":"https://drive.google.com/open?id=1exIN-55tUG1LFlgoHM582k8o8zy6H46f",

	# ? New link format 
	"Notebooks_NonDiv/BoatRouting_Time.ipynb":"https://drive.google.com/file/d/1T5sudWA6u23cG2mEqXF6wAXNpk-gAY1t/view?usp=sharing",
	"Notebooks_FMM/BoatRouting.ipynb":"https://drive.google.com/file/d/10xMu3f_0LcyBRu4Qymifz2I_MgWU6X2Y/view?usp=sharing",
	"Notebooks_FMM/TTI.ipynb":"https://drive.google.com/file/d/1bqWUzPHfEc3CEypMdIY-aGrTcEDfc6iL/view?usp=sharing",
	}


def Links(filename):
	if filename+".ipynb" not in google_drive_link:
		raise ValueError(f"File {filename} has no google drive link")
	links = {}
	for key,value in google_drive_link.items():
		link_prefix1 = 'https://drive.google.com/open?id='
		link_prefix2 = 'https://drive.google.com/file/d/'
		link_suffix2 = '/view?usp=sharing'
		if value.startswith(link_prefix1): 
			drive_id = value[len(link_prefix1):]
		elif value.startswith(link_prefix2) and value.endswith(link_suffix2):
			drive_id = value[len(link_prefix2):-len(link_suffix2)]
		else: raise ValueError('Invalid link format')

		colab_link = f"https://colab.research.google.com/notebook#fileId={drive_id}&offline=true&sandboxMode=true"
		subdir_,othername = os.path.split(key)
		if filename=='Summary':
			links[key]=colab_link
		else:
			if "/" in key: links[othername] = colab_link
			links["../"+key]=colab_link
	return links

def ToColab(filename,output_dir):
	if filename.startswith('Notebooks_GPU') and filename.endswith('_Repro'): return
	with open(filename+'.ipynb', encoding='utf8') as data_file:
		data=json.load(data_file)

	if 'Summary' not in filename:
		# Import the agd package from pip
		for cell in data['cells']:
			if (cell['cell_type']=='code' and len(cell['source'])>0 
				and cell['source'][0].startswith('import sys; sys.path.insert(0,"..")')):
				cell['source'] = ['pip install agd']
				# Do not forget to turn on GPU mode in Google Colab (R) parameters if necessary
				break
		else: raise ValueError(f"File {filename} does not import agd")

		#Use the GPU eikonal solver 
		UpdateConfig.EikonalGPU_config = True
		UpdateConfig.silent = True
		UpdateConfig(filename,data)

	links = Links(filename)
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

	with open(os.path.join(output_dir,filename+'_Colab.ipynb'),'w',encoding='utf8') as f:
		json.dump(data,f,ensure_ascii=False)

def Main(output_dir):
	for filename in ListNotebooks():
		ToColab(filename,output_dir)

if __name__ == "__main__":
	kwargs = {"output_dir":"/Users/mirebeau/Google Drive/AdaptiveGridDiscretizations_Colab"}
	Main(**kwargs)
	"""
	for key in sys.argv[1:]:
		prefix = '--output_dir='
		if key.startswith(prefix):
			output_dir = key[len(prefix):]
			break
	else: print("Missing output_dir=...")
	"""


