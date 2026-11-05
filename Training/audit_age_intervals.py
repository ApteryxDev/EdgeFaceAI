"""Audit calibration residuals by the assignment's fixed age/category strata.
The proposed conservative radius is fitted only on calibration rows. No test refitting.
"""
from pathlib import Path
import json,math
import torch,numpy as np
from student_model import StudentFaceModel
from train_student import Faces
from calibrate_and_evaluate import predict
ROOT=Path(__file__).resolve().parent;run=ROOT/'runs/cropped-A'
torch.set_num_threads(4)
model=StudentFaceModel(False).eval();model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True)['model'])
data=Faces(ROOT/'splits/calibration.jsonl');_,ages,gender,reference=predict(model,data,torch.device('cpu'));errors=(ages-reference).abs().numpy()
alpha=.2
strata={}
for name,mask in [('under18',reference<18),('adult-female',(reference>=18)&(reference<65)&(gender==0)),('adult-male',(reference>=18)&(reference<65)&(gender==1)),('older-female',(reference>=65)&(gender==0)),('older-male',(reference>=65)&(gender==1))]:
 e=errors[mask.numpy()];rank=math.ceil((len(e)+1)*(1-alpha))
 if rank>len(e):raise ValueError('Insufficient calibration rows for '+name)
 strata[name]={'examples':len(e),'radius_years':float(np.sort(e)[rank-1]),'coverage_original_radius':float((e<=8.773700714111328).mean())}
radius=max(v['radius_years'] for v in strata.values())
result={'method':'maximum of calibration-only 80% residual quantiles across prespecified reference-age/category strata; single conservative radius, no runtime reference label needed','target_coverage':.8,'original_radius_years':8.773700714111328,'candidate_radius_years':radius,'strata':strata,'limitations':'Wider intervals, not better age predictions. Finite-sample coverage requires exchangeability; correlated identities and deployment shift remain. Not tuned to the 40-image development results.'}
(run/'interval-strata-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
