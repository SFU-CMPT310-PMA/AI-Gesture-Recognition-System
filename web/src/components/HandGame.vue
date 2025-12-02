<template>
  <div class="wrap">
    <div class="camera-box">
      <video ref="videoRef" autoplay playsinline></video>
      <canvas ref="canvasRef"></canvas>
    </div>

    <div style="margin-top: 10px;">
      <button @click="startRound" :disabled="waitingForNextRound">
        {{ waitingForNextRound ? "Next round in progress..." : "Play" }}
      </button>
    </div>

    <div class="prediction-box">
      <h1>{{ prediction }}</h1>
    </div>
  </div>

  <div style="margin-top: 20px;">
    <button @click="showHistory = !showHistory">
      {{ showHistory ? "Hide History" : "Show History" }}
    </button>
  </div>

  <div v-if="showHistory" class="history-box">
    <h2>Game History</h2>

    <div v-for="item in history" :key="item.round" style="margin: 8px 0;">
      Round {{ item.round }}:
      You: {{ item.player }} |
      AI: {{ item.ai }}
      → {{ item.result }}
    </div>
  </div>

</template>

<script setup>


import { ref, onMounted } from "vue";
import { Hands, HAND_CONNECTIONS } from "@mediapipe/hands";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { Camera } from "@mediapipe/camera_utils";

const videoRef = ref(null);
const canvasRef = ref(null);
const prediction = ref("Waiting...");
const gameActive = ref(false);
const waitingForNextRound = ref(false);
const history = ref([]);
const showHistory = ref(false);
let roundCounter = 1;

let hands, camera;
// going for a minimalist scheme
const landmarksStyle = { color: "#FFFFFF", radius: 3 };
const connectionsStyle = { color: "#FFFFFF88", lineWidth: 2 };

onMounted(() => {
  hands = new Hands({
    locateFile: file =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });

  hands.onResults(onResults);

  camera = new Camera(videoRef.value, {
    onFrame: async () => {
      await hands.send({ image: videoRef.value });
    },
    width: 640,
    height: 480,
  });
  camera.start();
});

function toTitleCase(str) {
  if (str.toUpperCase() === "AI WINS!") return "AI Wins!";
  if (str.toUpperCase() === "YOU WIN!") return "You Win!";
  if (str.toUpperCase() === "TIE") return "Tie";

  // formtiing
  return str
    .toLowerCase()
    .replace(/\b\w/g, c => c.toUpperCase());
}


async function onResults(results) {
  const canvas = canvasRef.value;
  const ctx = canvas.getContext("2d");
  // set
  canvas.width = 640;
  canvas.height = 480;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks?.length) {
    const lm = results.multiHandLandmarks[0];
    // add connections
    drawConnectors(ctx, lm, HAND_CONNECTIONS, connectionsStyle);
    drawLandmarks(ctx, lm, landmarksStyle);

    const flattened = lm.map(p => [p.x, p.y, p.z]);
    sendToBackend(flattened);
  }
}

function startRound() {
  prediction.value = "Waiting for your move...";
  gameActive.value = true;
}


let roundCooldown = false;
async function sendToBackend(landmarks) {
  if (!gameActive.value || waitingForNextRound.value) return;
  if (roundCooldown) return;
  roundCooldown = true;
  setTimeout(() => (roundCooldown = false), 300);

  console.log("Sending to backend:", landmarks); // DEBUG

  try {
    const res = await fetch("http://localhost:8080/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks }),
    });

    if (!res.ok) {
      console.error("Backend returned error:", res.status);
      return;
    }

    const data = await res.json();
    // use this ofr debugging
    console.log("Received from backend:", data); 

    const player = toTitleCase(data.prediction);
    const ai = toTitleCase(aiMove());
    const result = toTitleCase(determineWinner(player.toUpperCase(), ai.toUpperCase()));


    prediction.value = `You: ${player} | AI: ${ai} | Result: ${result}`;
    history.value.push({
      round: roundCounter++,
      player,
      ai,
      result
    });

    waitingForNextRound.value = true;
    gameActive.value = false;


    setTimeout(() => {
      waitingForNextRound.value = false;
      
      prediction.value = "Click the play button to start the next round.";
    }, 3000);

  } catch (err) {
    console.error("FETCH ERROR:", err);
  }
}


// tester AI (bear with me)
function aiMove() {
  const moves = ["ROCK", "PAPER", "SCISSORS"];
  return moves[Math.floor(Math.random() * moves.length)];
}

function determineWinner(player, ai) {
  if (player === ai) return "TIE";

  if (
    (player === "ROCK" && ai === "SCISSORS") ||
    (player === "PAPER" && ai === "ROCK") ||
    (player === "SCISSORS" && ai === "PAPER")
  ) {
    return "YOU WIN!";
  }

  return "AI Wins!";
}


</script>

<style scoped>
.wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  margin: 0;
  padding: 0;
  /* background: rgb(255, 255, 255); */
}

.camera-box {
  position: relative;
  width: 640px;
  height: 480px;
}

video, canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 640px;
  height: 480px;
  border-radius: 8px;
  transform: scaleX(-1);
}

.prediction-box {
  font-size: 1rem;
  color: rgb(7, 64, 7);
  /* text-shadow: 0 0 15px rgba(134, 230, 110, 0.635); */
}

.history-box {
  margin: 20px auto;
  width: 450px;
  background: #f0f8f0;
  padding: 15px;
  border-radius: 8px;
  font-size: 1.1rem;
  box-shadow: 0 0 10px #ccc;
  text-align: center;
  color : rgb(7, 64, 7);
}

</style>
