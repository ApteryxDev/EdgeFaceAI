"""Train on training rows and select using validation only. Does not read calibration/test."""
import argparse,json,random,time,os,hashlib,platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from student_model import StudentFaceModel

class Faces(Dataset):
 def __init__(self,path,training=False,limit=None):
  self.rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
  self.rows=[r for r in self.rows if r['kind']=='cropped']
  if limit:self.rows=self.rows[:limit]
  common=[transforms.Resize((128,128))]
  if training:common += [transforms.RandomHorizontalFlip(),transforms.ColorJitter(.12,.12,.08,.02)]
  self.transform=transforms.Compose(common+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  # Corrupt images fail explicitly rather than being silently replaced during training.
  with Image.open(r['image_path']) as image:x=self.transform(image.convert('RGB'))
  return x,torch.tensor(r['gender'],dtype=torch.long),torch.tensor(float(r['age']))

def device_for(name):
 if name!='auto':return torch.device(name)
 return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
def evaluate(model,loader,device):
 model.eval();age_errors=[];correct=[0,0];total=[0,0]
 with torch.no_grad():
  for x,g,a in loader:
   logits,_,pred=model(x.to(device));guesses=logits.argmax(1).cpu();age_errors.extend((pred.cpu()-a).abs().tolist())
   for gender in [0,1]:
    mask=g==gender;total[gender]+=int(mask.sum());correct[gender]+=int(((guesses==g)&mask).sum())
 recalls=[c/n for c,n in zip(correct,total) if n]
 return {'age_mae':sum(age_errors)/len(age_errors),'gender_accuracy':sum(correct)/sum(total),
         'gender_balanced_accuracy':sum(recalls)/len(recalls),'examples':len(age_errors),'gender_counts':total}
def main():
 p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--epochs',type=int,default=20);p.add_argument('--freeze-epochs',type=int,default=2);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--device',default='auto');p.add_argument('--seed',type=int,default=20261108);p.add_argument('--initialization',choices=['imagenet','random'],default='imagenet');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True);os.environ.setdefault('TORCH_HOME',str(a.out.resolve()/'.torch-cache'))
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
 device=device_for(a.device);train=Faces(a.train,True,8 if a.smoke else None);val=Faces(a.validation,False,8 if a.smoke else None)
 if not len(train) or not len(val):raise ValueError('Both training and validation require cropped images.')
 train_groups={r['group_id'] for r in train.rows};val_groups={r['group_id'] for r in val.rows};assert not train_groups&val_groups
 model=StudentFaceModel(a.initialization=='imagenet').to(device)
 head=[*model.shared.parameters(),*model.gender.parameters(),*model.ages.parameters()]
 optimizer=torch.optim.AdamW([{'params':model.features.parameters(),'lr':a.lr*.1},{'params':head,'lr':a.lr}],weight_decay=1e-4)
 trainloader=DataLoader(train,batch_size=4 if a.smoke else a.batch_size,shuffle=True,num_workers=0)
 valloader=DataLoader(val,batch_size=4 if a.smoke else a.batch_size,num_workers=0)
 config={**vars(a),'device_used':str(device),'torch':torch.__version__,'python':platform.python_version(),
         'train_examples':len(train),'validation_examples':len(val),'age_bins':117,
         'selection_rule':'validation age MAE / 10 + (1 - gender balanced accuracy)',
         'scope':'SMOKE ONLY' if a.smoke else 'student training',
         'manifest_hashes':{str(path):hashlib.file_digest(path.open('rb'),'sha256').hexdigest() for path in [a.train,a.validation]}}
 config=json.loads(json.dumps(config,default=str))
 (a.out/'config.json').write_text(json.dumps(config,indent=2));best=float('inf');started=time.time()
 for epoch in range(1,(1 if a.smoke else a.epochs)+1):
  frozen=epoch<=a.freeze_epochs
  model.features.requires_grad_(not frozen);model.train()
  if frozen:model.features.eval()
  loss_total=0.;n=0
  for x,g,ages in trainloader:
   x,g,ages=x.to(device),g.to(device),ages.to(device);optimizer.zero_grad(set_to_none=True)
   gl,al,pred=model(x)
   targets=torch.exp(-.5*((model.age_values[None,:]-ages[:,None])/2.)**2);targets=targets/targets.sum(1,keepdim=True)
   age_loss=nn.functional.kl_div(al.log_softmax(1),targets,reduction='batchmean')
   loss=nn.functional.cross_entropy(gl,g)+age_loss
   loss.backward();optimizer.step();loss_total+=float(loss.detach())*len(x);n+=len(x)
  metrics=evaluate(model,valloader,device);row={'epoch':epoch,'frozen_backbone':frozen,'train_loss':loss_total/n,**metrics,'elapsed_seconds':time.time()-started,'scope':config['scope']}
  with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
  criterion=metrics['age_mae']/10+1-metrics['gender_balanced_accuracy']
  if criterion<best:
   best=criterion;torch.save({'model':model.cpu().state_dict(),'config':config,'epoch':epoch,'validation':metrics},a.out/('smoke.pt' if a.smoke else 'best.pt'));model.to(device)
 print('Completed',config['scope'],'No calibration or test rows were read.',flush=True)
if __name__=='__main__':main()
