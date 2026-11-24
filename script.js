import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");

let landmarker;
let stream;
let running = false;
let raf = 0;
let lastVideoTime = -1;
let calibration = null;
let calibrationSamples = [];
let calibrating = false;
let calibrationStarted = 0;
let poorSince = null;
let lastAlert = 0;
let lastVoice = "";

const modeSelect = $("viewMode");
const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const calibrateBtn = $("calibrateBtn");

const say = (title, detail, score = null, tip = "") => {
  $("statusText").textContent = title;
  $("statusDetail").textContent = detail;
  $("score").textContent = score === null ? "—" : Math.round(score);
  $("scoreBar").style.width = score === null ? "0%" : `${Math.max(0, Math.min(100, score))}%`;
  $("tipText").textContent = tip;
};

const avg = (values) => {
  const good = values.filter(Number.isFinite);
  return good.length ? good.reduce((a, b) => a + b, 0) / good.length : NaN;
};

function isVisible(point, threshold = 0.4) {
  return point && (point.visibility ?? 1) >= threshold;
}

function mid(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

async function loadModel() {
  $("message").textContent = "Loading the pose model…";
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
    minPoseDetectionConfidence: 0.45,
    minPosePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45
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
    $("placeholder").style.display = "none";
    $("cameraState").classList.add("on");
    $("cameraState").innerHTML = "<i></i> Camera on";
    startBtn.disabled = true;
    stopBtn.disabled = false;
    calibrateBtn.disabled = false;
    $("message").textContent = "Keep your face and shoulders in frame. Hips are not used.";
    resizeCanvas();
    loop();
  } catch (error) {
    console.error(error);
    startBtn.disabled = false;
    $("message").textContent = error.name === "NotAllowedError"
      ? "Camera permission was blocked. Allow it in your browser settings and try again."
      : "Could not start the camera. Use HTTPS or localhost and check camera access.";
  }
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(raf);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("placeholder").style.display = "flex";
  $("cameraState").classList.remove("on");
  $("cameraState").innerHTML = "<i></i> Not started";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  calibrateBtn.disabled = true;
  poorSince = null;
  say("Waiting for camera", "Start the camera and calibrate in a comfortable position.");
  $("message").textContent = "Camera stopped.";
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resizeCanvas);

function px(point) {
  return { x: (1 - point.x) * canvas.clientWidth, y: point.y * canvas.clientHeight };
}
function dot(point, color = "#f0a17d") {
  const p = px(point);
  ctx.beginPath();
  ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}
function line(a, b) {
  const p1 = px(a), p2 = px(b);
  ctx.beginPath();
  ctx.moveTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.strokeStyle = "#f0a17d";
  ctx.lineWidth = 3;
  ctx.stroke();
}

// Front view checks shoulder level and head centering.
// Side view checks how far the visible ear sits ahead of the shoulder.
// Neither mode reads hip landmarks.
function getMeasurements(lm, mode) {
  const ls = lm[11], rs = lm[12], nose = lm[0];
  if (!isVisible(ls, 0.4) || !isVisible(rs, 0.4)) return null;

  const shoulders = mid(ls, rs);
  const shoulderWidth = Math.max(0.04, Math.abs(ls.x - rs.x));
  const shoulderTilt = Math.abs(ls.y - rs.y) / shoulderWidth;

  if (mode === "front") {
    if (!isVisible(nose, 0.35)) return null;
    const headCenter = nose.x;
    return {
      primary: (headCenter - shoulders.x) / shoulderWidth,
      secondary: shoulderTilt,
      points: [ls, rs, nose],
      mode
    };
  }

  const ears = [lm[7], lm[8]].filter(e => isVisible(e, 0.35));
  if (!ears.length || !isVisible(nose, 0.35)) return null;
  const ear = ears.sort((a, b) => (b.visibility ?? 1) - (a.visibility ?? 1))[0];
  const shoulder = [ls, rs].sort((a, b) => Math.abs(a.x - ear.x) - Math.abs(b.x - ear.x))[0];

  // Direction from ear toward nose tells us which way the person is facing.
  const facing = Math.sign(nose.x - ear.x) || 1;
  const verticalScale = Math.max(0.06, Math.abs(ear.y - shoulder.y));
  const forwardHead = ((ear.x - shoulder.x) * facing) / verticalScale;

  return {
    primary: forwardHead,
    secondary: null,
    points: [ear, shoulder, nose],
    mode
  };
}

function drawMeasurements(m) {
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  m.points.forEach(p => dot(p));
  if (m.mode === "front") line(m.points[0], m.points[1]);
  else line(m.points[0], m.points[1]);
}

function beginCalibration() {
  calibrationSamples = [];
  calibrating = true;
  calibrationStarted = Date.now();
  poorSince = null;
  $("message").textContent = "Stay in a comfortable position for about two seconds…";
}

function analyze(lm) {
  const mode = modeSelect.value;
  const m = getMeasurements(lm, mode);

  if (!m) {
    poorSince = null;
    say("Move into view", "Show your face and both shoulders. In side view, keep the nearer ear visible.", null,
      "Move the camera a little farther back or improve the lighting.");
    return;
  }
  drawMeasurements(m);

  if (calibrating) {
    calibrationSamples.push({ primary: m.primary, secondary: m.secondary });
    const elapsed = Date.now() - calibrationStarted;
    say("Calibrating…", "Stay in your normal, comfortable position for a moment.");
    if (elapsed >= 2200 && calibrationSamples.length >= 8) {
      calibration = {
        mode,
        primary: avg(calibrationSamples.map(x => x.primary)),
        secondary: mode === "front" ? avg(calibrationSamples.map(x => x.secondary)) : null
      };
      calibrating = false;
      $("message").textContent = "Calibration saved. PostureX is comparing your position with this baseline.";
    }
    return;
  }

  if (!calibration || calibration.mode !== mode) {
    poorSince = null;
    say("Ready to calibrate", `You're in ${mode} view. Press Calibrate before starting the check.`, null,
      "Calibration gives PostureX a starting point for your normal posture.");
    return;
  }

  const primaryChange = Math.abs(m.primary - calibration.primary);
  const secondaryChange = mode === "front" ? Math.max(0, m.secondary - calibration.secondary) : 0;

  // Smooth, continuous score: even small deviations affect the score.
  const primaryPenalty = Math.min(55, primaryChange * (mode === "front" ? 125 : 42));
  const secondaryPenalty = Math.min(35, secondaryChange * 75);
  const score = Math.max(35, Math.min(100, 100 - primaryPenalty - secondaryPenalty));

  const primaryLimit = mode === "front" ? 0.075 : 0.16;
  const secondaryLimit = 0.12;
  const hasIssue = primaryChange > primaryLimit || secondaryChange > secondaryLimit;

  let title = "Looking steady";
  let detail = "Your position is close to the one you calibrated.";
  let tip = "Looks steady. Remember to take a short movement break now and then.";

  if (mode === "front" && secondaryChange > secondaryLimit) {
    title = "Shoulders look uneven";
    detail = "Your shoulder line has changed compared with your starting position.";
    tip = "Let your shoulders relax and check that you're not leaning to one side.";
  } else if (primaryChange > primaryLimit && mode === "front") {
    title = "Head has shifted";
    detail = "Your head position has moved sideways from your calibrated position.";
    tip = "Bring your head back to a comfortable, centred position.";
  } else if (primaryChange > primaryLimit && mode === "side") {
    title = "Head moving forward";
    detail = "Your ear is sitting farther forward relative to your shoulder than at calibration.";
    tip = "Gently bring your head back over your shoulder. Don't force your neck.";
  }

  say(title, detail, score, tip);
  if (hasIssue) {
    if (!poorSince) poorSince = Date.now();
    if (Date.now() - poorSince >= 5000) remind(title);
  } else {
    poorSince = null;
    lastVoice = "";
  }
}

function remind(title) {
  if (Date.now() - lastAlert < 12000) return;
  lastAlert = Date.now();

  if ($("soundToggle").checked) {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audio = new AudioCtx();
      const osc = audio.createOscillator();
      const gain = audio.createGain();
      osc.connect(gain);
      gain.connect(audio.destination);
      osc.frequency.value = 590;
      gain.gain.value = 0.055;
      osc.start();
      osc.stop(audio.currentTime + 0.18);
      osc.onended = () => audio.close();
    } catch (e) { console.warn("Sound unavailable", e); }
  }

  if ($("voiceToggle").checked && "speechSynthesis" in window && lastVoice !== title) {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance("Quick posture check. Relax your shoulders and get comfortable."));
    lastVoice = title;
  }
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    if (result.landmarks?.length) analyze(result.landmarks[0]);
    else {
      poorSince = null;
      say("No person detected", "Move into the camera frame so your face and shoulders can be seen.");
    }
  }
  raf = requestAnimationFrame(loop);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
calibrateBtn.addEventListener("click", beginCalibration);
modeSelect.addEventListener("change", () => {
  calibration = null;
  poorSince = null;
  if (running) say("View changed", "Press Calibrate again for this camera angle.");
});
window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach(t => t.stop());
});
