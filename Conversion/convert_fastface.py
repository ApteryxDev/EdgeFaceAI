from pathlib import Path
import json, hashlib
import numpy as np
import torch
from torch import nn
import coremltools as ct
import onnxruntime as ort
from PIL import Image
from age_gender_mobilenetv3 import AgeGenderMobileNetV3

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--source-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'Models/FastFaceAttributes.mlpackage')
args=parser.parse_args()
ROOT=args.source_dir.resolve()
DEST=args.output.resolve()
checkpoint = torch.load(ROOT/'best.pt', map_location='cpu', weights_only=True)
print('Checkpoint keys:', list(checkpoint), flush=True)
state = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint)))
base = AgeGenderMobileNetV3(variant='large', pretrained=False, age_bins=101).eval()
base.load_state_dict(state, strict=True)

class Wrapped(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
    def forward(self, rgb):
        out = self.model((rgb-self.mean)/self.std)
        return out['age'], torch.softmax(out['gender_logits'],dim=1)

wrapper = Wrapped(base).eval()
example = torch.zeros(1,3,128,128)
traced = torch.jit.trace(wrapper, example)
converted = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT32,
    inputs=[ct.ImageType(name='face', shape=example.shape, scale=1/255.0,
                         color_layout=ct.colorlayout.RGB)],
    outputs=[ct.TensorType(name='estimated_age'),ct.TensorType(name='gender_probabilities')],
    compute_units=ct.ComputeUnit.CPU_ONLY)
converted.author = 'FastFace / iFurySt; local Core ML conversion for IU project'
converted.license = 'Apache-2.0'
converted.short_description = 'FastFace Large 128: numeric age and dataset-defined female/male probabilities. ImageNet normalization embedded.'
converted.user_defined_metadata['source_github_commit'] = json.loads((ROOT/'repo-tree.json').read_text())['sha']
converted.user_defined_metadata['source_hf_commit'] = json.loads((ROOT/'hf-info.json').read_text())['sha']
converted.user_defined_metadata['input_contract'] = 'Aligned RGB 128x128; 1/255 scaling followed by embedded per-channel ImageNet mean/std.'
converted.user_defined_metadata['gender_order'] = '0=female,1=male'
converted.save(str(DEST))
print('Saved', DEST, flush=True)

# Numerical conversion checks use synthetic crops, not participant photos or accuracy labels.
rng = np.random.default_rng(20261106)
onnx = ort.InferenceSession(str(ROOT/'model_fp32.onnx'),providers=['CPUExecutionProvider'])
cases=[]
for name, pixels in [('black',np.zeros((128,128,3),dtype=np.uint8)),
                     ('white',np.full((128,128,3),255,dtype=np.uint8)),
                     ('random',rng.integers(0,256,(128,128,3),dtype=np.uint8)),
                     ('gradient',np.repeat(np.arange(128,dtype=np.uint8)[None,:,None]*2,128,axis=0).repeat(3,axis=2))]:
    rgb = pixels.astype(np.float32)/255
    tensor = torch.from_numpy(rgb.transpose(2,0,1)[None].copy())
    with torch.no_grad(): age,probs = wrapper(tensor)
    norm = ((rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None].copy()
    original = dict(zip([x.name for x in onnx.get_outputs()], onnx.run(None,{onnx.get_inputs()[0].name:norm})))
    actual = converted.predict({'face':Image.fromarray(pixels)})
    age_error=abs(float(age.numpy().ravel()[0])-float(np.asarray(actual['estimated_age']).ravel()[0]))
    probability_error=float(np.max(np.abs(probs.numpy()-actual['gender_probabilities'])))
    source_age_error=abs(float(age.numpy().ravel()[0])-float(np.asarray(original['age']).ravel()[0]))
    assert age_error<.01,(name,age_error)
    assert probability_error<.001,(name,probability_error)
    assert source_age_error<.01,(name,source_age_error)
    cases.append({'case':name,'coreml_age_absolute_error_years':age_error,
                  'coreml_probability_max_absolute_error':probability_error,
                  'onnx_age_absolute_error_years':source_age_error})
    print('PASS', cases[-1],flush=True)
report={'purpose':'Numerical synthetic-input conversion check; not face recognition accuracy',
        'torch':torch.__version__,'coremltools':ct.__version__,'cases':cases}
(ROOT/'conversion-check.json').write_text(json.dumps(report,indent=2))
