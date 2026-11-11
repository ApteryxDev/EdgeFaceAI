"""Prepare exact RGB pixels and PyTorch outputs for the iPhone conversion check."""
from pathlib import Path
import sys,json,hashlib
import torch,numpy as np
from PIL import Image
from torchvision.transforms import Resize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'Training'))
from student_model import StudentFaceModel
run=ROOT/'Training/runs/cropped-A';destination=ROOT/'Resources/ConversionFixtures';destination.mkdir(exist_ok=True)
model=StudentFaceModel(False).eval();model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True)['model'])
temperature=json.loads((run/'final-evaluation/calibration.json').read_text())['temperature']
rng=np.random.default_rng(20261108)
images=[('black',Image.fromarray(np.zeros((128,128,3),np.uint8))),('white',Image.fromarray(np.full((128,128,3),255,np.uint8))),('random',Image.fromarray(rng.integers(0,256,(128,128,3),dtype=np.uint8)))]
manifest=json.loads((ROOT/'Resources/UTKDevelopment/manifest.json').read_text())
for key in ['UTK-01','UTK-15']:
 row=next(x for x in manifest if x['id']==key)
 with Image.open(ROOT/'Resources/UTKDevelopment'/row['filename']) as image: images.append((key,Resize((128,128))(image.convert('RGB'))))
expected=[]
with torch.no_grad():
 for name,img in images:
  img.save(destination/(name+'.png'));x=torch.from_numpy(np.asarray(img).copy().transpose(2,0,1)[None]).float()/255
  x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
  logits,_,age=model(x)
  expected.append({'filename':name+'.png','age':float(age.item()),'probabilities':(logits/temperature).softmax(1)[0].tolist()})
(destination/'expected.json').write_text(json.dumps(expected,indent=2))
print('Prepared',len(expected),'identical-input fixtures, including a previously divergent older-face image.')
