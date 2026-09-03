<template>
  <div class="page">
    <header class="header">
      <h1 class="title">Rock Paper Scissors</h1>
      <p class="subtitle">Show your hand to the camera</p>
    </header>

    <main class="main">
      <div class="camera-wrapper">
        <div class="camera-box">
          <video ref="videoRef" autoplay playsinline></video>
          <canvas ref="canvasRef"></canvas>
          <transition name="countdown-fade">
            <div v-if="countdownValue !== null" class="countdown-overlay">
              <span :class="['countdown-number', countdownValue === 'GO!' ? 'go' : '']">
                {{ countdownValue }}
              </span>
            </div>
          </transition>
        </div>

        <div :class="['status-pill', statusClass]">
          {{ statusMessage }}
        </div>
      </div>

      <transition name="slide-up">
        <div v-if="lastResult" class="result-card">
          <div class="result-row">
            <div class="result-item">
              <span class="result-label">You</span>
              <span class="result-emoji">{{ moveEmoji(lastResult.player) }}</span>
              <span class="result-move">{{ lastResult.player }}</span>
            </div>
            <div class="result-vs">vs</div>
            <div class="result-item">
              <span class="result-label">AI</span>
              <span class="result-emoji">{{ moveEmoji(lastResult.ai) }}</span>
              <span class="result-move">{{ lastResult.ai }}</span>
            </div>
          </div>
          <div :class="['result-outcome', outcomeClass(lastResult.result)]">
            {{ lastResult.result }}
          </div>
        </div>
      </transition>

      <button
        class="play-btn"
        @click="startRound"
        :disabled="waitingForNextRound || countdownValue !== null"
      >
        <span v-if="waitingForNextRound || countdownValue !== null" class="btn-spinner"></span>
        {{ waitingForNextRound || countdownValue !== null ? "Wait..." : "Play" }}
      </button>

      <button class="history-btn" @click="showHistory = !showHistory">
        {{ showHistory ? "Hide History" : "Show History" }}
        <span v-if="history.length" class="history-badge">{{ history.length }}</span>
      </button>

      <transition name="slide-down">
        <div v-if="showHistory && history.length" class="history-box">
          <h2 class="history-title">History</h2>
          <div
            v-for="item in [...history].reverse()"
            :key="item.round"
            class="history-row"
          >
            <span class="history-round">#{{ item.round }}</span>
            <span class="history-detail">
              {{ moveEmoji(item.player) }} {{ item.player }}
              <span class="history-sep">vs</span>
              {{ moveEmoji(item.ai) }} {{ item.ai }}
            </span>
            <span :class="['history-result', outcomeClass(item.result)]">{{ item.result }}</span>
          </div>
        </div>
        <div v-else-if="showHistory && !history.length" class="history-box history-empty">
          No rounds played yet.
        </div>
      </transition>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from "vue";
import { Hands, HAND_CONNECTIONS } from "@mediapipe/hands";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { Camera } from "@mediapipe/camera_utils";

const videoRef = ref(null);
const canvasRef = ref(null);
const gameActive = ref(false);
const waitingForNextRound = ref(false);
const history = ref([]);
const showHistory = ref(false);
const countdownValue = ref(null); // null | 3 | 2 | 1 | "GO!"
const lastResult = ref(null);
let roundCounter = 1;
let roundInProgress = false;

let hands, camera;

const landmarksStyle = { color: "#FFFFFF", radius: 3 };
const connectionsStyle = { color: "#FFFFFF88", lineWidth: 2 };

const statusMessage = computed(() => {
  if (countdownValue.value !== null) return "Get ready...";
  if (waitingForNextRound.value) return "Round complete";
  if (gameActive.value) return "Show your hand!";
  return "Press Play to start";
});

const statusClass = computed(() => {
  if (gameActive.value && countdownValue.value === null && !waitingForNextRound.value) return "status-active";
  if (waitingForNextRound.value) return "status-done";
  return "";
});

onMounted(() => {
  hands = new Hands({
    locateFile: file =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
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

function startRound() {
  lastResult.value = null;
  gameActive.value = false;

  const steps = [3, 2, 1, "GO!"];
  let i = 0;

  function tick() {
    countdownValue.value = steps[i];
    i++;
    if (i < steps.length) {
      setTimeout(tick, 800);
    } else {
      // "GO!" shown for 600ms then capture begins
      setTimeout(() => {
        countdownValue.value = null;
        gameActive.value = true;
      }, 600);
    }
  }
  tick();
}

// mediaPipe results
async function onResults(results) {
  const canvas = canvasRef.value;
  const ctx = canvas.getContext("2d");
  canvas.width = 640;
  canvas.height = 480;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!results.multiHandLandmarks?.length) return;

  const lm = results.multiHandLandmarks[0];
  drawConnectors(ctx, lm, HAND_CONNECTIONS, connectionsStyle);
  drawLandmarks(ctx, lm, landmarksStyle);

  if (gameActive.value && !waitingForNextRound.value && !roundInProgress) {
    const flattened = lm.map(p => [p.x, p.y, p.z]);
    sendToBackend(flattened);
  }
}

async function sendToBackend(landmarks) {
  roundInProgress = true;

  try {
    const res = await fetch("http://localhost:8080/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks }),
    });

    const data = await res.json();

    const player = toTitleCase(data.prediction);
    const ai = toTitleCase(data.ai);
    const result = toTitleCase(determineWinner(player.toUpperCase(), ai.toUpperCase()));

    lastResult.value = { player, ai, result };
    history.value.push({ round: roundCounter++, player, ai, result });

    gameActive.value = false;
    waitingForNextRound.value = true;

    setTimeout(() => {
      roundInProgress = false;
      waitingForNextRound.value = false;
    }, 2000);
  } catch (err) {
    console.error("FETCH ERROR:", err);
    roundInProgress = false;
  }
}





function toTitleCase(str) {
  if (str.toUpperCase() === "AI WINS!") return "AI Wins!";
  if (str.toUpperCase() === "YOU WIN!") return "You Win!";
  if (str.toUpperCase() === "TIE") return "Tie";
  return str.toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
}

function determineWinner(player, ai) {
  if (player === ai) return "TIE";
  if (
    (player === "ROCK" && ai === "SCISSORS") ||
    (player === "PAPER" && ai === "ROCK") ||
    (player === "SCISSORS" && ai === "PAPER")
  ) return "YOU WIN!";
  return "AI Wins!";
}

function moveEmoji(move) {
  const m = (move || "").toUpperCase();
  if (m === "ROCK") return "✊";
  if (m === "PAPER") return "✋";
  if (m === "SCISSORS") return "✌️";
  return "❓";
}

function outcomeClass(result) {
  const r = (result || "").toLowerCase();
  if (r.includes("you win")) return "outcome-win";
  if (r.includes("ai wins") || r.includes("ai win")) return "outcome-loss";
  return "outcome-tie";
}
</script>

<style scoped>
.page {
  min-height: 100vh;
  background: #f8faf8;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0 1rem 3rem;
}

.header {
  text-align: center;
  padding: 2rem 0 1.25rem;
}

.title {
  font-size: 1.75rem;
  font-weight: 700;
  color: #1a3d1a;
  letter-spacing: -0.02em;
  margin: 0 0 0.25rem;
}

.subtitle {
  font-size: 0.9rem;
  color: #5a7a5a;
  margin: 0;
}

.main {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  width: 100%;
  max-width: 680px;
}
.camera-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.6rem;
  width: 100%;
}

.camera-box {
  position: relative;
  width: 640px;
  height: 480px;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 4px 24px rgba(0, 80, 0, 0.10);
  border: 2px solid #d4ead4;
  background: #000;
}

video,
canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 640px;
  height: 480px;
  border-radius: 14px;
  transform: scaleX(-1);
}

.countdown-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.45);
  border-radius: 14px;
  z-index: 10;
}

.countdown-number {
  font-size: 7rem;
  font-weight: 800;
  color: #fff;
  text-shadow: 0 0 40px rgba(100, 220, 100, 0.7), 0 2px 8px rgba(0,0,0,0.4);
  line-height: 1;
  animation: pop 0.25s ease-out;
}

.countdown-number.go {
  font-size: 4.5rem;
  color: #6dd86d;
  text-shadow: 0 0 40px rgba(100, 220, 100, 0.9);
}

@keyframes pop {
  0%   { transform: scale(0.6); opacity: 0.4; }
  60%  { transform: scale(1.08); }
  100% { transform: scale(1);   opacity: 1; }
}

.countdown-fade-enter-active,
.countdown-fade-leave-active {
  transition: opacity 0.2s ease;
}
.countdown-fade-enter-from,
.countdown-fade-leave-to {
  opacity: 0;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 1rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 500;
  background: #e8f3e8;
  color: #3a6a3a;
  border: 1px solid #c8e0c8;
  transition: background 0.3s, color 0.3s;
}

.status-pill::before {
  content: "";
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #98c898;
}

.status-active {
  background: #d0f0d0;
  color: #1e5a1e;
  border-color: #9eda9e;
}

.status-active::before {
  background: #3cb83c;
  box-shadow: 0 0 0 3px rgba(60, 184, 60, 0.2);
  animation: pulse 1.2s infinite;
}

.status-done {
  background: #f0f5f0;
  color: #7a9a7a;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.4; }
}

.result-card {
  width: 100%;
  background: #fff;
  border: 1px solid #d4ead4;
  border-radius: 14px;
  padding: 1.25rem 1.5rem;
  box-shadow: 0 2px 12px rgba(0, 80, 0, 0.06);
  text-align: center;
}

.result-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1.5rem;
  margin-bottom: 0.85rem;
}

.result-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.2rem;
}

.result-label {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #8aaa8a;
}

.result-emoji {
  font-size: 2.4rem;
  line-height: 1;
}

.result-move {
  font-size: 0.85rem;
  font-weight: 600;
  color: #2a4a2a;
}

.result-vs {
  font-size: 0.8rem;
  font-weight: 600;
  color: #aac8aa;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.result-outcome {
  font-size: 1.2rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.slide-up-enter-active {
  transition: all 0.35s ease-out;
}
.slide-up-enter-from {
  opacity: 0;
  transform: translateY(12px);
}
.outcome-win  { color: #1e7a1e; }
.outcome-loss { color: #a03030; }
.outcome-tie  { color: #7a7a3a; }

.play-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.7rem 2.8rem;
  font-size: 1rem;
  font-weight: 600;
  color: #fff;
  background: #2e7d2e;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: background 0.2s, transform 0.1s, box-shadow 0.2s;
  box-shadow: 0 2px 10px rgba(46, 125, 46, 0.3);
  letter-spacing: 0.02em;
}

.play-btn:hover:not(:disabled) {
  background: #256325;
  box-shadow: 0 4px 16px rgba(46, 125, 46, 0.4);
  transform: translateY(-1px);
}

.play-btn:active:not(:disabled) {
  transform: translateY(0);
}

.play-btn:disabled {
  background: #a0c8a0;
  cursor: not-allowed;
  box-shadow: none;
}

.btn-spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255,255,255,0.4);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.history-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.45rem 1.2rem;
  font-size: 0.85rem;
  font-weight: 500;
  color: #3a6a3a;
  background: transparent;
  border: 1px solid #c0dcc0;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s, border-color 0.2s;
}

.history-btn:hover {
  background: #eef7ee;
  border-color: #8abf8a;
}

.history-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 18px;
  padding: 0 4px;
  font-size: 0.7rem;
  font-weight: 700;
  background: #2e7d2e;
  color: #fff;
  border-radius: 999px;
}

.history-box {
  width: 100%;
  background: #fff;
  border: 1px solid #d4ead4;
  border-radius: 14px;
  padding: 1rem 1.25rem;
  box-shadow: 0 2px 12px rgba(0, 80, 0, 0.06);
}

.history-empty {
  text-align: center;
  color: #9ab89a;
  font-size: 0.9rem;
  padding: 1.5rem;
}

.history-title {
  font-size: 0.85rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #8aaa8a;
  margin: 0 0 0.75rem;
}

.history-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.45rem 0;
  border-bottom: 1px solid #f0f7f0;
  font-size: 0.88rem;
  color: #2a4a2a;
}

.history-row:last-child {
  border-bottom: none;
}

.history-round {
  font-size: 0.72rem;
  font-weight: 600;
  color: #9ab89a;
  min-width: 28px;
}

.history-detail {
  flex: 1;
  color: #3a5a3a;
}

.history-sep {
  color: #aac8aa;
  margin: 0 0.35rem;
  font-size: 0.78rem;
}

.history-result {
  font-weight: 600;
  font-size: 0.82rem;
}

.slide-down-enter-active,
.slide-down-leave-active {
  transition: all 0.3s ease;
  overflow: hidden;
}
.slide-down-enter-from,
.slide-down-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

@media (max-width: 680px) {
  .camera-box,
  video,
  canvas {
    width: 100%;
    height: auto;
    aspect-ratio: 4/3;
  }
}
</style>
