"""Export the frozen student checkpoint and calibrated category probabilities."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from torch import nn
import coremltools as ct
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Training'))
from student_model import StudentFaceModel
run=ROOT/'Training/runs/cropped-A'
checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
assert checkpoint['config']['scope']!='SMOKE ONLY'
cal=json.loads((run/'final-evaluation/calibration.json').read_text())
base=StudentFaceModel(False).eval();base.load_state_dict(checkpoint['model'])
class Wrapped(nn.Module):
 def __init__(self):
  super().__init__();self.model=base
  self.register_buffer('mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
  self.register_buffer('std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
 def forward(self,rgb):
  gender,_,age=self.model((rgb-self.mean)/self.std)
  return age,(gender/cal['temperature']).softmax(1)
wrapper=Wrapped().eval();example=torch.zeros(1,3,128,128)
converted=ct.convert(torch.jit.trace(wrapper,example),convert_to='mlprogram',minimum_deployment_target=ct.target.iOS17,compute_precision=ct.precision.FLOAT32,inputs=[ct.ImageType(name='face',shape=example.shape,scale=1/255.,color_layout=ct.colorlayout.RGB)],outputs=[ct.TensorType(name='estimated_age'),ct.TensorType(name='gender_probabilities')],compute_units=ct.ComputeUnit.CPU_ONLY)
sha=hashlib.file_digest((run/'best.pt').open('rb'),'sha256').hexdigest()
converted.author='IU EdgeFace AI student project with AI assistance'
converted.short_description='Student MobileNetV3 Small age and calibrated dataset gender categories'
converted.user_defined_metadata.update({'checkpoint_sha256':sha,'gender_order':'0=female,1=male','age_interval_radius_years':str(cal['age_interval_radius_years']),'target_coverage':str(cal['target_coverage']),'temperature':str(cal['temperature']),'input_contract':'RGB 128x128; embedded ImageNet normalization. Trained on resized UTK chips; live Vision alignment is an unvalidated domain shift.'})
converted.save(str(ROOT/'Models/StudentAttributes.mlpackage'))
cases=[]
rng=np.random.default_rng(20261108)
for name,pixels in [('black',np.zeros((128,128,3),np.uint8)),('white',np.full((128,128,3),255,np.uint8)),('random',rng.integers(0,256,(128,128,3),dtype=np.uint8))]:
 with torch.no_grad():age,prob=wrapper(torch.from_numpy(pixels.transpose(2,0,1)[None].copy()).float()/255)
 actual=converted.predict({'face':Image.fromarray(pixels)})
 ae=abs(float(age.item())-float(np.asarray(actual['estimated_age']).ravel()[0]));pe=float(np.max(np.abs(prob.numpy()-actual['gender_probabilities'])))
 assert ae<.01 and pe<.001,(ae,pe)
 cases.append({'case':name,'age_absolute_error':ae,'probability_max_error':pe})
report={'checkpoint_sha256':sha,'purpose':'Synthetic conversion parity, not face accuracy or device benchmark','cases':cases}
(run/'coreml-conversion-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
