from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

import torch
import os

path = "depth_anything_metric_depth_indoor"
pretrained_resource = f"{path}.pt"

model_name = "zoedepth"

dataset = "nyu"

overwrite = {"pretrained_resource": pretrained_resource}
config = get_config(model_name, "eval", dataset, **overwrite)
model = build_model(config)
model.eval()

output_path = f"{path}.onnx"
simplify_output_path = f"{path}_sim.onnx"

image_torch = torch.rand((1, 3, 518, 518))

torch.onnx.export(model,                                    # model being run
            image_torch,                                    # model input (or a tuple for multiple inputs)
            output_path,                                    # where to save the model (can be a file or file-like object)
            export_params=True,                             # store the trained parameter weights inside the model file
            opset_version=11,                               # the ONNX version to export the model to
            do_constant_folding=True,                       # whether to execute constant folding for optimization
            input_names = ['input'],                        # the model's input names
            output_names = ['output'],                      # the model's output names           
)       

os.system(f'onnxsim {output_path} {simplify_output_path}')