import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model/sign_model.keras")

mapping = {0: "rock", 1: "paper", 2: "scissors"}

def preprocess(landmarks):
    arr = np.array(landmarks).flatten()
    return np.expand_dims(arr, axis=0)

class RequestHandler(BaseHTTPRequestHandler):
    # need cors pre-flight
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_POST(self):
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        data = json.loads(body)
        landmarks = data.get("landmarks")
        x = preprocess(landmarks)

        pred = model.predict(x)
        idx = int(np.argmax(pred))
        label = mapping[idx]
        response = json.dumps({"prediction": label}).encode()

        # cors headers
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

def run():
    print("Server running at http://localhost:8080")
    server = HTTPServer(("0.0.0.0", 8080), RequestHandler)
    server.serve_forever()

if __name__ == "__main__":
    run()
