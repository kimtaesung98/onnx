#import onnxruntime as ort   

#session = ort.InferenceSession("/home/prlab/jetson_camera_project/onnx/model/model.onnx")

#print("success")
#print(ort.get_available_providers())

import torch
import torch.nn as nn

# 1. 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# 2. 모델 생성
model = SimpleModel()
model.eval()

# 3. 더미 입력
dummy_input = torch.randn(1, 10)

# 4. ONNX로 변환
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("model.onnx 생성 완료")
