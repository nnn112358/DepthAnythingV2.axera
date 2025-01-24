import cv2
import numpy as np
import axengine as axe
import threading
from queue import Queue
import time
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import json
import platform

#MODEL_PATH = "model/depth_anything_v2_vits_518x518.axmodel"
#MODEL_PATH = "depth_anything_v2_vits_378x378.onnx"
#MODEL_PATH = "depth_anything_v2_vits_224x224.onnx"
MODEL_PATH = "model/depth_anything_v2_vits_378x378.axmodel"

class InferenceEngine:
    def __init__(self, model_path):
        self.session = axe.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        input_shape = self.session.get_inputs()[0].shape
        self.target_size = (input_shape[1], input_shape[2])  # Get H,W from NCHW format
        print(self.target_size)
        print(input_shape)

    def preprocess_frame(self, frame):
        start_time = time.time()
        self.img_raw=frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.orig_h,self.orig_w = frame.shape[:2]

        img = cv2.resize(img, self.target_size)
      #  mean = np.array([123.675, 116.28, 103.53],dtype=np.float32).reshape(1,1,3)
      #  std = np.array([58.395, 57.12, 57.375],dtype=np.float32).reshape(1,1,3)
       # img = ((img - self.mean) / self.std).astype(np.float32)
       # img = np.transpose(img, (2, 0, 1))
       # img = np.expand_dims(img, axis=0)
       # img = (img-mean)/std 
       # img = img.transpose(2,0,1)
        img = img[None]
        self.preprocess_time = (time.time() - start_time) * 1000  # ms単位
        return img

    def run_inference(self, input_tensor):
        start_time = time.time()
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        self.inference_time = (time.time() - start_time) * 1000
        return output

    def postprocess_output(self, depth):
        start_time = time.time()

        depth = cv2.resize(depth[0, 0], (self.orig_w, self.orig_h))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        combined_result = cv2.hconcat([self.img_raw,  depth_color])
        #cv2.imshow("output-onnx", combined_result)

        self.postprocess_time = (time.time() - start_time) * 1000
        return combined_result

    def main_process(self, input_tensor):
        depth = self.run_inference(input_tensor)
        return self.postprocess_output(depth)


class CameraThread(threading.Thread):
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True
        self.camera_read_time = 0  # Add timing attribute
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            start_time = time.time()
            ret, frame = cap.read()
            self.camera_read_time = (time.time() - start_time) * 1000  # Convert to ms

            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put((frame, self.camera_read_time))
            time.sleep(0.05)

        cap.release()

    def stop(self):
        self.running = False


class InferenceThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, engine):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.engine = engine
        self.running = True

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame, camera_read_time = self.frame_queue.get()  # タプルから2つの値を取得
                input_tensor = self.engine.preprocess_frame(frame)
                predictions = self.engine.main_process(input_tensor)
                if self.result_queue.qsize() < 2:
                    self.result_queue.put((frame, predictions, camera_read_time))  # camera_read_timeを追加

            time.sleep(0.01)

    def stop(self):
        self.running = False


class DisplayThread(threading.Thread):
    def __init__(self, result_queue):
        super().__init__()
        self.result_queue = result_queue
        self.running = True
        self.latest_frame = None
        self.latest_predictions = None
        self.engine = None  # Add engine attribute

    def run(self):
        while self.running:
            if not self.result_queue.empty():
                frame, predictions, camera_read_time = self.result_queue.get()  # 3つの値を取得
                self.latest_frame = frame.copy()
                self.latest_predictions = predictions
                self.latest_camera_read_time = camera_read_time  # 保存

                # Process frame
                frame = self.process_frame(frame.copy(), predictions)

                if platform.machine() == 'x86_64':
                    cv2.imshow("Camera Feed", frame)
#                    cv2.imshow("latest_frame", self.latest_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
            time.sleep(0.03)
        cv2.destroyAllWindows()

    def process_frame(self, frame, predictions):
        # Draw predictions

        # Add timing information
        timings = [
            f"Camera read: {self.latest_camera_read_time:.1f}ms",
            f"Preprocess: {self.engine.preprocess_time:.1f}ms",
            f"Inference: {self.engine.inference_time:.1f}ms",
            f"Postprocess: {self.engine.postprocess_time:.1f}ms"
        ]
        for i, timing in enumerate(timings):
            y_pos = predictions.shape[0] - 30 * (len(timings) - i)
            cv2.putText(predictions, timing, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return predictions

    def get_latest_frame(self):
        if self.latest_frame is not None:
            return self.process_frame(self.latest_frame.copy(), self.latest_predictions)
        return None

    def stop(self):
        self.running = False


display_thread = None  # グローバル変数として保持


def generate_frames():
    while True:
        if display_thread and display_thread.latest_frame is not None:
            frame = display_thread.get_latest_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

app = FastAPI()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Camera Stream</title>
    </head>
    <body>
        <h1>Depth Anything v2 Stream</h1>
        <img src="/video_feed">
    </body>
    </html>
    """

def main():
    global display_thread

    engine = InferenceEngine(MODEL_PATH)
    frame_queue = Queue(maxsize=10)
    result_queue = Queue(maxsize=10)

    camera_thread = CameraThread(frame_queue)
    inference_thread = InferenceThread(frame_queue, result_queue, engine)
    display_thread = DisplayThread(result_queue)

    display_thread.engine = engine

    camera_thread.start()
    inference_thread.start()
    display_thread.start()

    def shutdown_handler():
        print("\nShutting down gracefully...")
        camera_thread.stop()
        inference_thread.stop()
        display_thread.stop()

        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                pass
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except:
                pass

        camera_thread.join(timeout=1)
        inference_thread.join(timeout=1)
        display_thread.join(timeout=1)

    import uvicorn
    import atexit
    atexit.register(shutdown_handler)

    try:
        uvicorn.run(app, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        shutdown_handler()

if __name__ == "__main__":
    main()