"""Fit uncertainty using calibration rows; test evaluation requires an explicit flag."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from student_model import StudentFaceModel
from train_student import Faces,device_for

def predict(model,data,device):
 logits=[];ages=[];labels=[];references=[]
 with torch.no_grad():
  for x,g,a in DataLoader(data,batch_size=32,num_workers=0):
   gl,_,age=model(x.to(device));logits.append(gl.cpu());ages.append(age.cpu());labels.append(g);references.append(a)
 return torch.cat(logits),torch.cat(ages),torch.cat(labels),torch.cat(references)
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--test',type=Path);p.add_argument('--evaluate-test',action='store_true');p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='auto');p.add_argument('--alpha',type=float,default=.2);a=p.parse_args()
 if not 0<a.alpha<1:raise ValueError('alpha must lie between 0 and 1')
 checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 if checkpoint['config'].get('scope')=='SMOKE ONLY':raise ValueError('Smoke checkpoint is not a trained candidate for evaluation.')
 model=StudentFaceModel(imagenet=False);model.load_state_dict(checkpoint['model']);device=device_for(a.device);model.to(device).eval()
 calibration=Faces(a.calibration);gl,age,gender,reference=predict(model,calibration,device)
 log_temperature=torch.zeros(1,requires_grad=True);optimizer=torch.optim.LBFGS([log_temperature],lr=.1,max_iter=50)
 def closure():
  optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(gl/log_temperature.exp().clamp(.05,20),gender);loss.backward();return loss
 optimizer.step(closure);temperature=float(log_temperature.exp().clamp(.05,20).detach())
 errors=(age-reference).abs().numpy();rank=math.ceil((len(errors)+1)*(1-a.alpha))
 if rank>len(errors):raise ValueError('Too few calibration samples for a finite requested conformal interval.')
 radius=float(np.sort(errors)[rank-1]);a.out.mkdir(parents=True,exist_ok=True)
 result={'checkpoint':str(a.checkpoint),'calibration_examples':len(errors),'temperature':temperature,
         'age_interval_radius_years':radius,'target_coverage':1-a.alpha,
         'meaning':'Approximation against noisy dataset age labels, not verified biological age; empirical test coverage must be reported.'}
 (a.out/'calibration.json').write_text(json.dumps(result,indent=2))
 if a.evaluate_test:
  if a.test is None:raise ValueError('--test is required with --evaluate-test')
  test=Faces(a.test);assert not {r['group_id'] for r in calibration.rows}&{r['group_id'] for r in test.rows}
  logits,age,gender,reference=predict(model,test,device);probs=(logits/temperature).softmax(1);pred=probs.argmax(1);errors=(age-reference).abs()
  output=[]
  for i,r in enumerate(test.rows):
   output.append({'source_key':r['source_key'],'sha256':r['sha256'],'reference_age':float(reference[i]),'predicted_age':float(age[i]),'absolute_age_error':float(errors[i]),'reference_gender':int(gender[i]),'predicted_gender':int(pred[i]),'gender_score':float(probs[i].max()),'age_interval_low':max(0,float(age[i])-radius),'age_interval_high':min(116,float(age[i])+radius),'covered':bool(errors[i]<=radius)})
  with (a.out/'test-predictions.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
  result['test']={'examples':len(test),'age_mae':float(errors.mean()),'within_5_years':float((errors<=5).float().mean()),'gender_accuracy':float((pred==gender).float().mean()),'age_interval_coverage':float((errors<=radius).float().mean())}
  (a.out/'test-metrics.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
