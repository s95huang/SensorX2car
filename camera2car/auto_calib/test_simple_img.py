import torch
import cv2
import numpy as np
from models import build_model
import math
from config import cfg
from util.misc import NestedTensor
import numpy.linalg as LA
from pylsd import lsd

def compute_horizon(pred_hl, pred_vp, img_sz):
    hl_left = pred_vp[1] + (pred_vp[0] + img_sz[1] / 2) * math.tan(pred_hl)
    hl_right = pred_vp[1] + (pred_vp[0] - img_sz[1] / 2) * math.tan(pred_hl)
    return hl_left, hl_right
    
def filter_length(segs, min_line_length=10):
    lengths = LA.norm(segs[:,2:4] - segs[:,:2], axis=1)
    segs = segs[lengths > min_line_length]
    return segs[:,:4]

def normalize_segs(segs, pp, rho):    
    pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)    
    return rho*(segs - pp)    

def normalize_safe_np(v, axis=-1, eps=1e-6):
    de = LA.norm(v, axis=axis, keepdims=True)
    de = np.maximum(de, eps)
    return v/de

def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device, dtype=torch.float32) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32).to(device)
             for k, v in t.items()} for t in data]


def segs2lines_np(segs):
    ones = np.ones(len(segs))
    ones = np.expand_dims(ones, axis=-1)
    p1 = np.concatenate([segs[:,:2], ones], axis=-1)
    p2 = np.concatenate([segs[:,2:], ones], axis=-1)
    lines = np.cross(p1, p2)
    return normalize_safe_np(lines)

def sample_segs_np(segs, num_sample, use_prob=True):    
    num_segs = len(segs)
    sampled_segs = np.zeros([num_sample, 4], dtype=np.float32)
    mask = np.zeros([num_sample, 1], dtype=np.float32)
    if num_sample > num_segs:
        sampled_segs[:num_segs] = segs
        mask[:num_segs] = np.ones([num_segs, 1], dtype=np.float32)
    else:    
        lengths = LA.norm(segs[:,2:] - segs[:,:2], axis=-1)
        prob = lengths/np.sum(lengths)        
        idxs = np.random.choice(segs.shape[0], num_sample, replace=True, p=prob)
        sampled_segs = segs[idxs]
        mask = np.ones([num_sample, 1], dtype=np.float32)
    return sampled_segs, mask

def generate_lines_and_mask(image, num_lines=512, min_line_length=10):
    height = image.shape[0]
    width = image.shape[1]

    # detect line segments
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segs = lsd(gray, scale=0.8)
    segs = filter_length(segs, min_line_length)
    num_segs = len(segs)
    
    # normalize segment coordinates
    pp = (width/2, height/2)
    rho = 2.0/np.minimum(width,height)
    segs = normalize_segs(segs, pp=pp, rho=rho)

    # sample fixed number of line segments
    sampled_segs, line_mask = sample_segs_np(segs, num_lines)
    lines = segs2lines_np(sampled_segs)

    return lines, line_mask

# load the model
cfg.merge_from_file('./config-files/ctrlc.yaml')
device = torch.device(cfg.DEVICE)
    
model, _ = build_model(cfg)
model.to(device)
checkpoint = torch.load('./logs/checkpoint.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model = model.eval()

# load the image
img = cv2.imread('./pic/000016.png')

img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# get image original size
org_sz = (img.shape[0], img.shape[1])
img = cv2.resize(img_copy, (512, 512))
img_resized = img.copy()

# convert to tensor
img = torch.from_numpy(img).float()
img = img.permute(2, 0, 1)

# create nested tensor
img_tensor = img.unsqueeze(0)  # add batch dimension
mask = torch.zeros((1, img.shape[1], img.shape[2]))  # create mask with zeros
mask[:, :org_sz[0], :org_sz[1]] = 1  # set padded pixels to 1
samples = {'tensor': img_tensor, 'mask': mask}
nested_tensor = NestedTensor(samples['tensor'], samples['mask'])

# create the extra_samples
lines, line_mask = generate_lines_and_mask(img_resized)
extra_samples = {'lines': torch.from_numpy(lines).unsqueeze(0), 'line_mask': torch.from_numpy(line_mask).unsqueeze(0)}

# run the forward pass
with torch.no_grad():


# torch.Size([1, 512, 3])
# torch.Size([1, 512, 1])
    
# torch.Size([1, 3, 512, 512])
# torch.Size([1, 512, 512])

    # print(nested_tensor.tensors.shape)
    print(extra_samples['lines'].shape)
    print(extra_samples['line_mask'].shape)

    # check nested_tensor shape
    print(nested_tensor.tensors.shape)
    print(nested_tensor.mask.shape)

    nested_tensor = nested_tensor.to(device)
    extra_samples = to_device(extra_samples, device)

    # ensure nested_tensor is on the same device as extra_samples
    nested_tensor = nested_tensor.to(extra_samples['lines'].device)

    # ensure extra_samples and nested_tensor are all using float32
    nested_tensor = nested_tensor.to(torch.float32)
    outputs = model(nested_tensor, extra_samples)


# get the prediction
pred_vp = outputs['pred_vp'].to('cpu')[0].numpy()
pred_hl = outputs['pred_hl'].to('cpu')[0].numpy()

# convert to the original image size
img_sz = org_sz
pp = (img_sz[1]/2, img_sz[0]/2)
rho = 2.0/np.minimum(img_sz[0],img_sz[1])

pred_vp[0] = pred_vp[0] / pred_vp[2] / rho
pred_vp[1] = pred_vp[1] / pred_vp[2] / rho

hl_left, hl_right = compute_horizon(pred_hl, pred_vp, img_sz)

print(pred_vp)
print(pred_hl)

print(hl_left, hl_right)

# output
