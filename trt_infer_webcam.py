import tensorrt as trt
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as tv2
import numpy as np
import cv2
import time

engname = "cityscapes_trt.engine"

logger = trt.Logger()
runtime = trt.Runtime(logger)
with open(engname, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

inputs = []
outputs = []
allocs = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)

    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    print(name, mode, dtype, shape)

    allocation = torch.zeros(
        list(shape), dtype=torch.float32, device=torch.device("cuda")
    )
    allocs.append(allocation)
    binding = {
        "index": i,
        "name": name,
        "dtype": np.dtype(trt.nptype(dtype)),
        "shape": [shape],
        "allocation": allocation.data_ptr(),
    }
    if mode == trt.TensorIOMode.INPUT:
        inputs.append(binding)
    elif mode == trt.TensorIOMode.OUTPUT:
        outputs.append(binding)
    else:
        pass

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera. Try change the device ID.")
    exit(1)

preproc = tv2.Compose(
    [
        tv2.CenterCrop(512),
        tv2.ToDtype(torch.float32, scale=True),
        tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

for ix in range(10):
    t_begin = time.time()
    err, frame = cap.read()
    cv2.imwrite(f"{ix}_frame.jpg", frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).permute(2, 0, 1)
    frame = preproc(frame)
    print(frame.mean(), frame.std())

    allocs[0].copy_(frame)
    tensors = [x.data_ptr() for x in allocs]
    context.execute_v2(tensors)

    # After this, seg is 512x512 grayscale image
    seg = F.softmax(allocs[1], dim=1).max(dim=1)[1].squeeze()

    t_end = time.time()
    print("Processed in {:.4f}s".format(t_end - t_begin))

    # Dumb multiply by 10 to get back into the 0-255 range
    seg = 10 * seg.to(torch.uint8).cpu().numpy()
    cv2.imwrite(f"{ix}_segment.jpg", seg)

    # cv2.imshow('segment', seg)
    # if cv2.pollKey() > -1:
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     break
