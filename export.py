from depth_anything.dpt import DepthAnything
import torch
import os
import sys

class NormLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x / x.amax(dim=(1, 2), keepdim=True)

variants = ['l', 'b', 's'] #'l' #'b' #'s'

for variant in variants:
    path = f"depth_anything_vit{variant}14"

    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnything.from_pretrained(f"LiheYoung/{path}").eval()
    norm_model = torch.nn.Sequential(model, NormLayer())

    output_path = f"{path}.onnx"
    simplify_output_path = f"{path}_sim.onnx"

    image_torch = torch.rand((1, 3, 518, 518))

    output_torch = norm_model(image_torch)
    
    torch.onnx.export(norm_model,                               # model being run
                image_torch,                                    # model input (or a tuple for multiple inputs)
                output_path,                                    # where to save the model (can be a file or file-like object)
                export_params=True,                             # store the trained parameter weights inside the model file
                opset_version=11,                               # the ONNX version to export the model to
                do_constant_folding=True,                       # whether to execute constant folding for optimization
                input_names = ['input'],                        # the model's input names
                output_names = ['output'],                      # the model's output names           
    )       

    os.system(f'onnxsim {output_path} {simplify_output_path}')