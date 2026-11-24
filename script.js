import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const calibrateBtn = document.getElementById("calibrateBtn");
const placeholder = document.getElementById("placeholder");
const cameraState = document.getElementById("cameraState");
const message = document.getElementById("message");
const statusText = document.getElementById("statusText");
const statusDetail = document.getElementById("statusDetail");
const scoreEl = document.getElementById("score");
const scoreBar = document.getElementById("scoreBar");
const tipText = document.getElementById("tipText");
const statusCard = document.getElementById("statusCard");

let landmarker = null;
let stream = null;
let running = false;
let rafId = null;
let lastVideoTime = -1;
let badSince = null;
let lastReminder = 0;
let calibration = null;
let calibrationSamples = [];
let calibrating = false;
let lastSpoken = "";
let lastStatus = "WAITING";

const LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12;
const LEFT_EAR = 7, RIGHT_EAR = 8;
const NOSE = 0;

function setStatus(title, detail, score, tip) {
  statusText.textContent = title;
  statusDetail.textContent = detail;
  scoreEl.textContent = score == null ? "--" : Math.round(score);
  scoreBar.style.width = score == null ? "0%" : `${Math.max(0, Math.min(100, score))}%`;
  tipText.textContent = tip;
  lastStatus = title;
}

function visible(point, minimum = 0.45) {
  return point && (point.visibility ?? 1) >= minimum;
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

async function loadModel() {
  message.textContent = "Loading the posture model…";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
}

async function startCamera() {
  try {
    startBtn.disabled = true;
    if (!landmarker) await loadModel();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    running = true;
    placeholder.style.display = "none";
    cameraState.textContent = "CAMERA ON";
    cameraState.style.background = "#eff8e7";
    cameraState.style.color = "#648c35";
    startBtn.disabled = true;
    stopBtn.disabled = false;
    calibrateBtn.disabled = false;
    message.textContent = "Keep your face and both shoulders in view. Hips are not needed.";
    resizeCanvas();
    detectFrame();
  } catch (error) {
    console.error(error);
    startBtn.disabled = false;
    message.textContent = error.name === "NotAllowedError"
      ? "Camera permission was blocked. Allow camera access in your browser settings and try again."
      : "Could not start the camera. Use HTTPS or localhost and check that another app is not using it.";
  }
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  placeholder.style.display = "block";
  cameraState.textContent = "CAMERA OFF";
  cameraState.style.background = "";
  cameraState.style.color = "";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  calibrateBtn.disabled = true;
  badSince = null;
  message.textContent = "Camera stopped.";
  setStatus("Waiting", "Start the camera to begin checking your upper-body posture.", null, "Sit comfortably, then start a calibration.");
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

window.addEventListener("resize", resizeCanvas);

function drawPoint(point, color = "#b9f36b") {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.beginPath();
  ctx.arc((1 - point.x) * w, point.y * h, 5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawLine(a, b, color = "#b9f36b") {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.beginPath();
  ctx.moveTo((1 - a.x) * w, a.y * h);
  ctx.lineTo((1 - b.x) * w, b.y * h);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.stroke();
}

function analyze(landmarks) {
  const leftShoulder = landmarks[LEFT_SHOULDER];
  const rightShoulder = landmarks[RIGHT_SHOULDER];
  const nose = landmarks[NOSE];

  // Hips are deliberately not checked. Only the face and shoulder area are required.
  if (!visible(leftShoulder, 0.4) || !visible(rightShoulder, 0.4)) {
    badSince = null;
    setStatus("Move into view", "Please show your face and both shoulders. Your hips can stay outside the frame.", null, "Adjust the camera so your upper body is visible.");
    return;
  }

  const shoulders = midpoint(leftShoulder, rightShoulder);
  const shoulderWidth = Math.max(0.05, Math.abs(leftShoulder.x - rightShoulder.x));
  const shoulderTilt = Math.abs(leftShoulder.y - rightShoulder.y) / shoulderWidth;

  if (visible(nose, 0.35)) {
    drawLine(nose, shoulders, "#b9f36b");
    drawPoint(nose);
  }
  drawPoint(leftShoulder);
  drawPoint(rightShoulder);
  drawLine(leftShoulder, rightShoulder);

  const earLeft = landmarks[LEFT_EAR], earRight = landmarks[RIGHT_EAR];
  let headOffset = null;
  if (visible(earLeft, 0.35) && visible(earRight, 0.35)) {
    const ears = midpoint(earLeft, earRight);
    headOffset = Math.abs(ears.x - shoulders.x) / shoulderWidth;
  } else if (visible(nose, 0.35)) {
    headOffset = Math.abs(nose.x - shoulders.x) / shoulderWidth;
  }

  const features = { shoulderTilt, headOffset };
  if (calibrating) {
    calibrationSamples.push(features);
    const elapsed = Date.now() - calibrationStartedAt;
    message.textContent = `Calibrating… keep a comfortable upright position (${Math.min(2, elapsed / 1000).toFixed(1)}s)`;
    if (elapsed >= 2000 && calibrationSamples.length >= 10) {
      calibration = {
        shoulderTilt: average(calibrationSamples.map(s => s.shoulderTilt)),
        headOffset: average(calibrationSamples.map(s => s.headOffset).filter(Number.isFinite))
      };
      calibrating = false;
      message.textContent = "Calibration saved. Posture checks are now relative to your starting position.";
      calibrationSamples = [];
    }
    setStatus("Calibrating", "Stay in your comfortable upright position for a moment.", null, "Calibration helps the system compare your posture with your own baseline.");
    return;
  }

  if (!calibration) {
    setStatus("Ready to calibrate", "Your upper body is visible. Sit comfortably and press Calibrate posture.", null, "Calibration takes about two seconds and uses only your face and shoulders.");
    return;
  }

  const tiltChange = Math.max(0, shoulderTilt - calibration.shoulderTilt);
  const headChange = headOffset == null || !Number.isFinite(calibration.headOffset)
    ? 0 : Math.max(0, headOffset - calibration.headOffset);

  let issue = "";
  let detail = "Your upper-body position is close to your calibrated baseline.";
  let tip = "Keep your shoulders relaxed and take regular movement breaks.";
  let penalty = 0;

  if (headChange > 0.10) {
    issue = "Head position";
    detail = "Your head appears to have shifted forward or sideways compared with calibration.";
    tip = "Gently bring your head back over your shoulders without forcing your neck.";
    penalty += Math.min(35, (headChange - 0.10) * 180);
  }
  if (tiltChange > 0.10) {
    issue = issue ? "Adjust your posture" : "Shoulder tilt";
    detail = "Your shoulders look less level than they did during calibration.";
    tip = "Relax both shoulders and sit evenly. Avoid forcing them into position.";
    penalty += Math.min(30, (tiltChange - 0.10) * 100);
  }

  const score = Math.max(40, Math.min(100, 100 - penalty));
  if (issue) {
    setStatus(issue, detail, score, tip);
    if (!badSince) badSince = Date.now();
    if (Date.now() - badSince > 5000) remind(issue);
  } else {
    badSince = null;
    setStatus("Looking steady", detail, score, tip);
  }
}

let calibrationStartedAt = 0;
function average(values) {
  const valid = values.filter(v => Number.isFinite(v));
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : NaN;
}

function remind(issue) {
  if (Date.now() - lastReminder < 12000) return;
  lastReminder = Date.now();
  if (document.getElementById("soundToggle").checked) {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audio = new AudioContextClass();
      const oscillator = audio.createOscillator();
      const gain = audio.createGain();
      oscillator.connect(gain);
      gain.connect(audio.destination);
      oscillator.frequency.value = 660;
      gain.gain.value = 0.07;
      oscillator.start();
      oscillator.stop(audio.currentTime + 0.16);
      oscillator.onended = () => audio.close();
    } catch (e) { console.warn("Audio reminder unavailable", e); }
  }
  if (document.getElementById("voiceToggle").checked && "speechSynthesis" in window && lastSpoken !== issue) {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance("Check your posture and relax your shoulders."));
    lastSpoken = issue;
  }
}

function detectFrame() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    const result = landmarker.detectForVideo(video, performance.now());
    if (result.landmarks && result.landmarks.length) {
      analyze(result.landmarks[0]);
    } else {
      badSince = null;
      setStatus("No person detected", "Move back into the camera frame so your face and shoulders are visible.", null, "Try improving the lighting and keeping your upper body in view.");
    }
  }
  rafId = requestAnimationFrame(detectFrame);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
calibrateBtn.addEventListener("click", () => {
  calibrationSamples = [];
  calibrating = true;
  calibrationStartedAt = Date.now();
  badSince = null;
  message.textContent = "Sit in your normal comfortable upright position…";
});
window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
});
