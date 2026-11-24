POSTUREAI — UPPER BODY VERSION

WHAT CHANGED
- The detection logic does not require hip landmarks (23 and 24).
- It uses the face and shoulder landmarks instead.
- Keep your face and both shoulders visible in the camera.
- The system compares your current position with a short personal calibration.
- It can give reminders after 5 seconds when it notices a change from that baseline.

HOW TO RUN
1. Put index.html, style.css and script.js in the same folder.
2. Open the folder in VS Code.
3. Run index.html using the Live Server extension, or deploy the files to GitHub Pages.
4. Allow camera permission.
5. Sit comfortably and click "Calibrate posture".

IMPORTANT
This is an educational posture-estimation project, not a medical device.
Without hips or a side-view reference, it cannot reliably assess full-spine,
lower-back, or whole-body alignment. The score is an approximate visual
indicator based on face/shoulder landmarks and the user's calibration.

If GPU model loading fails on a device, change delegate: "GPU" to delegate: "CPU"
in script.js and try again. Camera access generally requires HTTPS or localhost.
