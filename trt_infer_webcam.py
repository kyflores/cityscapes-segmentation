import os
import tensorrt as trt
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as tv2
import numpy as np
import cv2
import time


def load_trt_engine(engname: str):
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

    print(f"Loaded engine {engname}")
    return {
        "engine": engine,
        "context": context,
        "allocs": allocs,
        "inputs": inputs,
        "outputs": outputs,
    }


def webcam_infer(engname: str, cap_device):
    cap = cv2.VideoCapture(cap_device)
    # Try to select a format with a higher res. Check `v4l2-ctl --list-formats-ext`
    # to see what the camera supports. If you need a different shape the height/width
    # must be changed when exporting ONNX
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 544)

    if not cap.isOpened():
        print("Could not open camera. Try change the device ID.")
        exit(1)

    trt_vars = load_trt_engine(engname)

    preproc = tv2.Compose(
        [
            tv2.CenterCrop(512),
            tv2.ToDtype(torch.float16, scale=True),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    os.makedirs("segment", exist_ok=True)
    for ix in range(1000):
        ix_pad = str(ix).zfill(4)
        status, frame = cap.read()
        if not status:
            print(f"During capture, status returned {status}")
            break
        cv2.imwrite(f"segment/frame_{ix_pad}.jpg", frame)

        t_begin = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).cuda()
        frame = preproc(frame)
        t_preproc = time.time()
        # print(frame.mean(), frame.std())

        trt_vars["allocs"][0].copy_(frame)
        tensors = [x.data_ptr() for x in trt_vars["allocs"]]
        trt_vars["context"].execute_v2(tensors)

        # After this, seg is 512x512 grayscale image
        seg = F.softmax(trt_vars["allocs"][1], dim=1).max(dim=1)[1].squeeze()

        t_end = time.time()
        print(
            "Preproc in {:.4f}s, infer in {:.4f}".format(
                t_preproc - t_begin, t_end - t_preproc
            )
        )

        # Dumb multiply by 10 to get back into the 0-255 range
        seg = 10 * seg.to(torch.uint8).cpu().numpy()
        cv2.imwrite(f"segment/segment_{ix_pad}.jpg", seg)

        # cv2.imshow('segment', seg)
        # if cv2.pollKey() > -1:
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     break


if __name__ == "__main__":
    if not os.path.isfile("/etc/nv_tegra_release"):
        print("This script must be run on a tegra device.")
        exit(1)

    engname = "cityscapes_trt.engine"
    webcam_infer(engname, "/home/kyle/Downloads/00.mp4")
