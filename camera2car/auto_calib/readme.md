# Testing log

## 2023 02 23

* created nested tensor from tensor and mask
* created extra_samples from lines and line_mask

* ensure that input tensor shape is correct

Current error log:
"""
/usr/bin/python3 /mnt/0c39e9c4-f324-420d-a1e9-f20a41d147a8/personal_repos/auto_calibration/SensorX2car/camera2car/auto_calib/test_simple_img.py
torch.Size([1, 512, 3])
torch.Size([1, 512, 1])
torch.Size([1, 3, 512, 512])
torch.Size([1, 512, 512])
Traceback (most recent call last):
  File "/mnt/0c39e9c4-f324-420d-a1e9-f20a41d147a8/personal_repos/auto_calibration/SensorX2car/camera2car/auto_calib/test_simple_img.py", line 142, in <module>
    outputs = model(nested_tensor, extra_samples)
  File "/home/s95huang/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/mnt/0c39e9c4-f324-420d-a1e9-f20a41d147a8/personal_repos/auto_calibration/SensorX2car/camera2car/auto_calib/models/ctrlc.py", line 73, in forward
    tgt=self.input_line_proj(lines), 
  File "/home/s95huang/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/s95huang/.local/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 must have the same dtype

"""