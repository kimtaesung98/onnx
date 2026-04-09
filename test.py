import onnxruntime as ort   

session = ort.InferenceSession("/home/prlab/jetson_camera_project/onnx/model/model.onnx")

print("success")
#print(ort.get_available_providers())