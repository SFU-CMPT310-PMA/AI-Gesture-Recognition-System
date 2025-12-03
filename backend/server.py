import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import tensorflow as tf
from ai_opponent.ai_opponent_original import BayesianAIOpponent
# Load gesture model
model = tf.keras.models.load_model("backend/model/sign_model.keras")
mapping = {0: "ROCK", 1: "PAPER", 2: "SCISSORS"}

# Initialize Bayesian AI opponent
ai = BayesianAIOpponent()

def preprocess(landmarks):
   arr = np.array(landmarks).flatten()
   return np.expand_dims(arr, axis=0)

class RequestHandler(BaseHTTPRequestHandler):
   def do_OPTIONS(self):
       # cors headers
       self.send_response(200)
       self.send_header("Access-Control-Allow-Origin", "*")
       self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
       self.send_header("Access-Control-Allow-Headers", "Content-Type")
       self.end_headers()

   def do_POST(self):
       # handles post
       if self.path != "/predict":
           self.send_response(404)
           self.end_headers()
           return
       
       # Read request body
       content_len = int(self.headers.get("Content-Length", 0))
       body = self.rfile.read(content_len)
       data = json.loads(body)


       landmarks = data.get("landmarks")
       if landmarks is None:
           self.send_response(400)
           self.end_headers()
           return
       # Run gesture model
       x = preprocess(landmarks)
       pred = model.predict(x)
       idx = int(np.argmax(pred))
       player_label = mapping[idx]

       # -------- BAYESIAN AI LOGIC --------
       predicted_user_next = ai.predictNextGesture()
       ai_gesture_idx = ai.updateAINextGesture(predicted_user_next)
       ai_label = mapping[ai_gesture_idx]

       # Update AI model with the user's *actual* gesture
       ai.update(idx)
       # ----------------------------------
       # Prepare response
       response_data = {
           "prediction": player_label,  # player's current move
           "ai": ai_label               # AI’s counter move
       }
       response_json = json.dumps(response_data).encode()

       # Send response
       self.send_response(200)
       self.send_header("Content-Type", "application/json")
       self.send_header("Access-Control-Allow-Origin", "*")
       self.send_header("Content-Length", str(len(response_json)))
       self.end_headers()
       self.wfile.write(response_json)

def run():
   print("Bayesian AI server running at http://localhost:8080")
   server = HTTPServer(("0.0.0.0", 8080), RequestHandler)
   server.serve_forever()

if __name__ == "__main__":
   run()