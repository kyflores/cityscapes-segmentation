# Claude made (most of) this :)

import os
import sys
import signal
import cv2
import threading
import queue
import time
import argparse
from flask import Flask, Response, render_template_string

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as tv2
import tensorrt as trt

import dataset

app = Flask(__name__)

# Global frame queue
frame_queue = queue.Queue(maxsize=2)  # Small buffer to keep latency low

shutdown_flag = threading.Event()


def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) to gracefully shut down"""
    print("\nShutting down...")
    shutdown_flag.set()

    time.sleep(1.0)
    sys.exit(0)


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


def camera_thread(engname: str, cap_device=0):
    trt_vars = load_trt_engine(engname)

    preproc = tv2.Compose(
        [
            # tv2.CenterCrop(512),
            tv2.Resize((1024, 2048)),
            tv2.ToDtype(torch.float16, scale=True),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cap = cv2.VideoCapture(cap_device)  # Use webcam, change to your camera source

    # Try to select a format with a specific res. Check `v4l2-ctl --list-formats-ext`
    # to see what the camera supports. If you need a different shape the height/width
    # must be changed when exporting ONNX
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HisOpenedEIGHT, 544)

    if not cap.isOpened():
        print("Could not open capture source. Try change the device ID or path.")
        sys.exit(1)

    while not shutdown_flag.is_set():
        ret, frame = cap.read()
        if ret:
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
            # seg = F.softmax(trt_vars["allocs"][1], dim=1).max(dim=1)[1].squeeze()
            thresh = 0.5
            max_val, max_idx = F.softmax(trt_vars["allocs"][1], dim=1).max(dim=1)
            seg = torch.ones_like(max_idx) * dataset.CityScapesDataset.BACKGROUND
            thresholded_mask = max_val > thresh
            seg[thresholded_mask] = max_idx[thresholded_mask]

            print(seg.shape)

            t_end = time.time()
            print(
                "Preproc in {:.4f}s, infer in {:.4f}".format(
                    t_preproc - t_begin, t_end - t_preproc
                )
            )

            # Dumb multiply by 10 to get back into the 0-255 range
            seg = 10 * seg.squeeze().to(torch.uint8).cpu().numpy()

            # Put frame in queue, drop old frames if queue is full
            if not frame_queue.full():
                frame_queue.put(seg, timeout=1.0)
            else:
                try:
                    frame_queue.get_nowait()  # Remove old frame
                    frame_queue.put(seg, timeout=1.0)  # Add new frame
                except queue.Empty:
                    pass
    cap.release()
    print("Infer thread done.")


def generate_frames():
    """Generator function to yield JPEG frames"""
    while True:
        try:
            # Get frame from queue with timeout
            frame = frame_queue.get(timeout=1.0)

            # Encode frame as JPEG (compressed format for bandwidth)
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

            # Yield frame in multipart format
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        except queue.Empty:
            # If no frame available, yield empty response to keep connection alive
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n\r\n")


@app.route("/")
def index():
    """Main page with video feed"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Feed</title>
    </head>
    <body>
        <h1>Live Video Feed</h1>
        <img src="/video_feed" width="2048" height="1024">
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/video_feed")
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    # Set up signal handler for cancelling the camera thread.
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser("Segmentation Inferer")
    parser.add_argument(
        "-p",
        "--port",
        help="Port to start webserver on",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "-e",
        "--engine",
        help="Path to TensorRT Engine file.",
        default="cityscapes_trt.engine",
    )
    parser.add_argument(
        "-c",
        "--capture",
        help="Capture ID to pass on to cv2.VideoCapture. Can be a number, or path to video file.",
        default="0",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="Threshold for segmentation post processing. Pixels with scores below threshold are assigned to the background class.",
        type=float,
        default=0.5,
    )
    opt = parser.parse_args()

    try:
        opt.capture = int(opt.capture)
    except ValueError:
        print("Assuming capture is a path...")

    # Start camera thread.
    camera_worker = threading.Thread(
        target=camera_thread, daemon=True, args=(opt.engine, opt.capture)
    )
    camera_worker.start()

    # Start Flask app
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
