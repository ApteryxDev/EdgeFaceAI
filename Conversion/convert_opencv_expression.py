"""Translate the pinned OpenCV ONNX graph into Torch, verify, then export Core ML.
Only Conv/PRelu/Add/Flatten/Gemm operators present in this release are accepted.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import onnx,onnxruntime as ort
import torch
from torch import nn
import torch.nn.functional as F
import coremltools as ct
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'Models/OpenCVExpression';graph=onnx.load(source/'model.onnx')
torch.set_num_threads(4)
labels=['angry','disgust','fear','happy','neutral','sad','surprise']
class Graph(nn.Module):
 def __init__(self):
  super().__init__();self.names={}
  for i,init in enumerate(graph.graph.initializer):
   self.register_buffer('weight_'+str(i),torch.from_numpy(onnx.numpy_helper.to_array(init).copy()));self.names[init.name]='weight_'+str(i)
  self.nodes=[(n.op_type,list(n.input),list(n.output),{a.name:onnx.helper.get_attribute_value(a) for a in n.attribute}) for n in graph.graph.node]
 def forward(self,x):
  values={name:getattr(self,attr) for name,attr in self.names.items()};values['data']=x
  for op,inputs,outputs,a in self.nodes:
   v=[values[k] for k in inputs]
   if op=='Conv':
    pads=a.get('pads',[0,0,0,0]);assert pads[:2]==pads[2:]
    y=F.conv2d(v[0],v[1],v[2] if len(v)>2 else None,stride=a.get('strides',[1,1]),padding=pads[:2],dilation=a.get('dilations',[1,1]),groups=a.get('group',1))
   elif op=='PRelu':y=F.prelu(v[0],v[1].reshape(-1))
   elif op=='Add':y=v[0]+v[1]
   elif op=='Flatten':y=torch.flatten(v[0],start_dim=a.get('axis',1))
   elif op=='Gemm':
    assert a.get('transA',0)==0 and a.get('transB',0)==1
    y=a.get('alpha',1.)*F.linear(v[0],v[1])+a.get('beta',1.)*v[2]
   else:raise ValueError('Unsupported ONNX operation '+op)
   assert len(outputs)==1;values[outputs[0]]=y
  return values['label']
class Wrapped(nn.Module):
 def __init__(self):super().__init__();self.model=Graph()
 def forward(self,rgb):return self.model(rgb*2.-1.).softmax(1)
wrapper=Wrapped().eval();rng=np.random.default_rng(20261108);session=ort.InferenceSession(str(source/'model.onnx'),providers=['CPUExecutionProvider'])
fixtures=ROOT/'Resources/ConversionFixtures';fixtures.mkdir(exist_ok=True);cases=[];expected=[]
for name,pixels in [('expression-black',np.zeros((112,112,3),np.uint8)),('expression-white',np.full((112,112,3),255,np.uint8)),('expression-random',rng.integers(0,256,(112,112,3),dtype=np.uint8))]:
 x=torch.from_numpy(pixels.transpose(2,0,1)[None].copy()).float()/255
 with torch.no_grad():prob=wrapper(x).numpy()
 logits=session.run(None,{'data':x.numpy()*2.-1.})[0];native=np.exp(logits-logits.max(1,keepdims=True));native/=native.sum(1,keepdims=True)
 error=float(abs(prob-native).max());assert error<.0001,(name,error)
 cases.append({'case':name,'onnx_torch_probability_max_error':error});Image.fromarray(pixels).save(fixtures/(name+'.png'))
 expected.append({'filename':name+'.png','probabilities':prob[0].tolist()})
model=ct.convert(torch.jit.trace(wrapper,torch.zeros(1,3,112,112)),convert_to='mlprogram',minimum_deployment_target=ct.target.iOS17,compute_precision=ct.precision.FLOAT32,inputs=[ct.ImageType(name='face',shape=(1,3,112,112),scale=1/255.,color_layout=ct.colorlayout.RGB)],outputs=[ct.TensorType(name='expression_probabilities')],classifier_config=ct.ClassifierConfig(labels,predicted_feature_name='classLabel',predicted_probabilities_output='expression_probabilities'),skip_model_load=True)
model.author='OpenCV Zoo Progressive Teacher; local Core ML conversion with AI assistance';model.license='Apache-2.0';model.short_description='MobileFaceNet expression candidate; normalized RGB112, seven dataset categories'
model.user_defined_metadata.update({'source_url':'https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition','onnx_sha256':hashlib.file_digest((source/'model.onnx').open('rb'),'sha256').hexdigest(),'preprocessing':'RGB112; x/127.5-1; five-point standard112 alignment. Vision landmarks approximate source detector points.','labels':','.join(labels)})
model.save(str(ROOT/'Models/OpenCVExpression.mlpackage'))
(source/'conversion-check.json').write_text(json.dumps({'purpose':'ONNX to Torch numerical equivalence; on-device Core ML parity remains pending','cases':cases,'onnx_sha256':model.user_defined_metadata['onnx_sha256']},indent=2))
(fixtures/'expression-expected.json').write_text(json.dumps(expected,indent=2))
print('Saved candidate model; classifier outputs:',model.get_spec().description)
