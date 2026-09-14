/**
 * PS26052 ANC Unified Tactical Command Center — Frontend Controller
 * Handles audio scenario loading, live microphone stream capture,
 * backend neural cascade invocation, dual-channel spectrogram rendering,
 * and synchronized listening station playback.
 */

// Application State
let currentSourceType = 'preset'; // 'preset' | 'upload' | 'mic'
let availablePresets = [];
let uploadedAudioBase64 = null;
let recordedAudioBase64 = null;
let lastProcessResult = null;

// Microphone Recording State
let mediaStream = null;
let audioContext = null;
let audioInputNode = null;
let analyserNode = null;
let recorderNode = null;
let isRecording = false;
let recordStartTime = 0;
let recordTimerInterval = null;
let recordedBuffers = [];
let recordedLength = 0;
let animFrameId = null;

// Audio A/B State
let activeABChannel = 'output'; // 'input' | 'output'

document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  // 1. Draw initial realistic spectrogram heatmaps matching target design
  drawInitialSpectrograms();

  // 2. Setup mobile sidebar toggle
  setupMobileToggle();

  // 3. Load hardware platform and available presets
  await Promise.all([loadHardwareStatus(), loadPresets()]);
}

/**
 * Setup mobile hamburger menu toggle
 */
function setupMobileToggle() {
  const toggleBtn = document.querySelector('.menu-toggle-btn');
  const sidebar = document.querySelector('.sidebar');
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }
}

/**
 * Fetch hardware and recommended models
 */
async function loadHardwareStatus() {
  try {
    const res = await fetch('/api/hardware');
    const data = await res.json();

    const platBadge = document.getElementById('platformBadge');
    if (platBadge && data.os) {
      platBadge.textContent = `${data.os}`;
    }

    if (data.recommended_model) {
      const modelSelect = document.getElementById('modelSelect');
      if (modelSelect && data.models && data.models[data.recommended_model]) {
        modelSelect.value = data.recommended_model;
      }
    }
  } catch (err) {
    console.warn('Hardware status load error:', err);
  }
}

/**
 * Fetch preset defence audio scenarios
 */
async function loadPresets() {
  const select = document.getElementById('presetSelect');
  if (!select) return;

  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    availablePresets = data.presets || [];

    select.innerHTML = '';
    if (availablePresets.length === 0) {
      select.innerHTML = '<option value="" disabled>No preset audio assets found</option>';
      return;
    }

    availablePresets.forEach((preset, index) => {
      const opt = document.createElement('option');
      opt.value = preset.id;
      opt.textContent = preset.name;
      if (index === 0) opt.selected = true;
      select.appendChild(opt);
    });

    onPresetSelected();
  } catch (err) {
    select.innerHTML = '<option value="" disabled>Error loading presets</option>';
    console.error('Preset loading failed:', err);
  }
}

/**
 * Switch Audio Input Source Tabs (Preset vs Upload vs Mic)
 */
function switchSourceTab(type) {
  currentSourceType = type;

  document.querySelectorAll('.modal-tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(c => c.classList.remove('active'));

  const badge = document.getElementById('selectedSourceBadge');

  if (type === 'preset') {
    document.getElementById('tabPresetBtn').classList.add('active');
    document.getElementById('tabPresetContent').classList.add('active');
    if (badge) badge.textContent = 'PRESET';
  } else if (type === 'upload') {
    document.getElementById('tabUploadBtn').classList.add('active');
    document.getElementById('tabUploadContent').classList.add('active');
    if (badge) badge.textContent = 'FILE UPLOAD';
  } else if (type === 'mic') {
    document.getElementById('tabMicBtn').classList.add('active');
    document.getElementById('tabMicContent').classList.add('active');
    if (badge) badge.textContent = 'LIVE MIC';
    setupOscilloscopeIdle();
  }
}

/**
 * Preset selection change
 */
function onPresetSelected() {
  const select = document.getElementById('presetSelect');
  if (!select) return;
  const selectedId = select.value;
  const preset = availablePresets.find(p => p.id === selectedId);

  const infoText = document.getElementById('presetDetailsText');
  if (preset && infoText) {
    const refTag = preset.has_clean_reference
      ? 'Ground-Truth Clean Speech Reference available (SI-SNR & STOI will be evaluated).'
      : 'Mono Noisy Sample.';
    infoText.textContent = `Acoustic Scenario: ${preset.category.toUpperCase()} noise at ${preset.snr}. ${refTag}`;
  }

  const pillText = document.getElementById('currentPresetPillText');
  if (preset && pillText) {
    pillText.textContent = `${preset.category.toUpperCase()} Noise (SNR ${preset.snr})`;
  }
}

/**
 * Navigation handlers
 */
function setActiveNav(element) {
  document.querySelectorAll('.sidebar-nav .nav-item').forEach(el => el.classList.remove('active'));
  if (element) {
    element.classList.add('active');
  }
}

function scrollToSection(sectionId) {
  const elem = document.getElementById(sectionId);
  if (elem) {
    elem.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

/**
 * Source Selection Modal controls
 */
function openSourceModal() {
  const modal = document.getElementById('sourceModal');
  if (modal) {
    modal.classList.add('open');
  }
}

function closeSourceModal(event) {
  if (event && event.target !== event.currentTarget) {
    return;
  }
  const modal = document.getElementById('sourceModal');
  if (modal) {
    modal.classList.remove('open');
  }
}

/**
 * Drag & Drop / File Input Handler
 */
function handleFileSelected(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    uploadedAudioBase64 = event.target.result;
    document.getElementById('loadedFileName').textContent = file.name;
    document.getElementById('loadedFileSize').textContent = `${(file.size / 1024).toFixed(1)} KB`;
    document.getElementById('fileLoadedBanner').style.display = 'flex';

    const pillText = document.getElementById('currentPresetPillText');
    if (pillText) {
      pillText.textContent = `File: ${file.name.length > 18 ? file.name.substring(0, 16) + '...' : file.name}`;
    }
  };
  reader.readAsDataURL(file);
}

// Drag & Drop Setup
const dropzone = document.getElementById('fileDropzone');
if (dropzone) {
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.style.borderColor = '#38bdf8';
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.style.borderColor = 'rgba(56, 189, 248, 0.35)';
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    if (file) {
      const input = document.getElementById('audioFileInput');
      input.files = dt.files;
      handleFileSelected({ target: { files: [file] } });
    }
  });
}

/**
 * Pipeline Mode Selection (Hybrid, AI-only, ANC-only)
 */
function onModeChanged() {
  const mode = document.getElementById('pipelineModeSelect').value;
  const modelGroup = document.getElementById('modelSelectGroup');
  const filterGroup = document.getElementById('filterTapsGroup');
  const stepGroup = document.getElementById('stepSizeGroup');

  if (mode === 'ai_only') {
    modelGroup.style.display = 'flex';
    filterGroup.style.display = 'none';
    stepGroup.style.display = 'none';
  } else if (mode === 'anc_only') {
    modelGroup.style.display = 'none';
    filterGroup.style.display = 'flex';
    stepGroup.style.display = 'flex';
  } else {
    // Hybrid
    modelGroup.style.display = 'flex';
    filterGroup.style.display = 'flex';
    stepGroup.style.display = 'flex';
  }
}

/**
 * Microphone Oscilloscope Idle Display
 */
function setupOscilloscopeIdle() {
  const canvas = document.getElementById('micOscilloscopeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#03060f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
}

/**
 * Live Browser Microphone Recording
 */
async function toggleMicRecording() {
  const btn = document.getElementById('micRecordBtn');
  const btnText = document.getElementById('micRecordBtnText');
  const timer = document.getElementById('recordTimer');

  if (!isRecording) {
    // Start Recording
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      });
    } catch (err) {
      alert(`Could not access microphone: ${err.message}. Please allow microphone permissions in your browser.`);
      return;
    }

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    audioInputNode = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 512;

    const bufferSize = 4096;
    recorderNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
    recordedBuffers = [];
    recordedLength = 0;

    recorderNode.onaudioprocess = (e) => {
      if (!isRecording) return;
      const inputBuffer = e.inputBuffer.getChannelData(0);
      recordedBuffers.push(new Float32Array(inputBuffer));
      recordedLength += inputBuffer.length;
    };

    audioInputNode.connect(analyserNode);
    audioInputNode.connect(recorderNode);
    recorderNode.connect(audioContext.destination);

    isRecording = true;
    recordStartTime = Date.now();
    btn.classList.add('recording');
    btnText.textContent = 'Stop & Process Audio';

    recordTimerInterval = setInterval(() => {
      const elapsedMs = Date.now() - recordStartTime;
      const secs = Math.floor(elapsedMs / 1000);
      const dec = Math.floor((elapsedMs % 1000) / 100);
      const mm = String(Math.floor(secs / 60)).padStart(2, '0');
      const ss = String(secs % 60).padStart(2, '0');
      timer.textContent = `${mm}:${ss}.${dec}`;
    }, 100);

    drawOscilloscope();
  } else {
    // Stop Recording
    isRecording = false;
    clearInterval(recordTimerInterval);
    cancelAnimationFrame(animFrameId);

    btn.classList.remove('recording');
    btnText.textContent = 'Start Recording';

    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
    }
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close();
    }

    const merged = new Float32Array(recordedLength);
    let offset = 0;
    for (let i = 0; i < recordedBuffers.length; i++) {
      merged.set(recordedBuffers[i], offset);
      offset += recordedBuffers[i].length;
    }

    recordedAudioBase64 = encodeWAV(merged, 16000);
    timer.textContent = 'Ready to Process';

    const pillText = document.getElementById('currentPresetPillText');
    if (pillText) {
      pillText.textContent = 'Live Mic Recording';
    }

    closeSourceModal();
    runProcessingPipeline();
  }
}

/**
 * Draw animated oscilloscope from live microphone
 */
function drawOscilloscope() {
  if (!isRecording) return;
  animFrameId = requestAnimationFrame(drawOscilloscope);

  const canvas = document.getElementById('micOscilloscopeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyserNode.getByteTimeDomainData(dataArray);

  ctx.fillStyle = '#03060f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 2;
  ctx.strokeStyle = '#38bdf8';
  ctx.beginPath();

  const sliceWidth = canvas.width * 1.0 / bufferLength;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const v = dataArray[i] / 128.0;
    const y = v * (canvas.height / 2);

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
    x += sliceWidth;
  }

  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
}

/**
 * Client-Side WAV Encoder (Float32 to 16-bit mono WAV Data URI)
 */
function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return 'data:audio/wav;base64,' + btoa(binary);
}

/**
 * Execute Full Noise Cancellation Pipeline
 */
async function runProcessingPipeline() {
  const executeBtn = document.getElementById('executeBtn');
  const spinner = document.getElementById('executeSpinner');
  const icon = document.getElementById('executeIcon');
  const btnText = document.getElementById('executeBtnText');
  const statusBadge = document.getElementById('systemStatusBadge');

  const mode = document.getElementById('pipelineModeSelect').value;
  const modelName = document.getElementById('modelSelect').value;
  const filterTaps = parseInt(document.getElementById('filterTapsSelect').value, 10);
  const stepSize = parseFloat(document.getElementById('stepSizeSelect').value);

  const payload = {
    mode: mode,
    model_name: modelName,
    filter_length: filterTaps,
    step_size: stepSize,
  };

  if (currentSourceType === 'preset') {
    payload.preset_id = document.getElementById('presetSelect').value;
  } else if (currentSourceType === 'upload') {
    if (!uploadedAudioBase64) {
      openSourceModal();
      switchSourceTab('upload');
      alert('Please select or upload an audio file first.');
      return;
    }
    payload.audio_base64 = uploadedAudioBase64;
  } else if (currentSourceType === 'mic') {
    if (!recordedAudioBase64) {
      openSourceModal();
      switchSourceTab('mic');
      alert('Please record a live voice sample first.');
      return;
    }
    payload.audio_base64 = recordedAudioBase64;
  }

  // Set Loading UI
  executeBtn.disabled = true;
  spinner.style.display = 'inline-block';
  icon.style.display = 'none';
  btnText.textContent = 'Processing...';
  if (statusBadge) {
    statusBadge.textContent = 'Processing Audio';
    statusBadge.style.color = '#38bdf8';
  }

  try {
    const res = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!data.success) {
      alert(`Processing error: ${data.error || 'Unknown error'}`);
      return;
    }

    lastProcessResult = data;
    renderResults(data);

  } catch (err) {
    alert(`Server communication error: ${err.message}`);
    console.error('Process error:', err);
  } finally {
    executeBtn.disabled = false;
    spinner.style.display = 'none';
    icon.style.display = 'inline-block';
    btnText.textContent = 'Execute';
    if (statusBadge) {
      statusBadge.textContent = 'System Ready';
      statusBadge.style.color = '#f8fafc';
    }
  }
}

/**
 * Render Audio, Spectrograms, and Metrics HUD
 */
function renderResults(data) {
  // 1. Audio Players
  const inPlayer = document.getElementById('inputAudioPlayer');
  const refPlayer = document.getElementById('refAudioPlayer');
  const outPlayer = document.getElementById('outputAudioPlayer');
  const refBox = document.getElementById('refChannelBox');
  const dlBtn = document.getElementById('downloadEnhancedBtn');
  const abToggleBtn = document.getElementById('abToggleBtn');

  if (data.input_audio_base64 && inPlayer) {
    inPlayer.src = data.input_audio_base64;
  }

  if (data.reference_audio_base64 && refPlayer) {
    refPlayer.src = data.reference_audio_base64;
    if (refBox) refBox.style.display = 'flex';
  } else if (refBox) {
    refBox.style.display = 'none';
  }

  if (data.enhanced_audio_base64 && outPlayer) {
    outPlayer.src = data.enhanced_audio_base64;
    if (dlBtn) {
      dlBtn.href = data.enhanced_audio_base64;
      dlBtn.style.display = 'inline-flex';
    }
    if (abToggleBtn) abToggleBtn.disabled = false;
  }

  // 2. Spectrograms
  const inSpecImg = document.getElementById('inputSpectrogramImg');
  const outSpecImg = document.getElementById('outputSpectrogramImg');
  const inCanvas = document.getElementById('inputSpecCanvas');
  const outCanvas = document.getElementById('outputSpecCanvas');
  const inPh = document.getElementById('inputSpecPlaceholder');
  const outPh = document.getElementById('outputSpecPlaceholder');

  if (data.input_spectrogram_base64 && inSpecImg) {
    inSpecImg.src = data.input_spectrogram_base64;
    inSpecImg.style.display = 'block';
    if (inCanvas) inCanvas.style.display = 'none';
    if (inPh) inPh.style.display = 'none';
  }

  if (data.output_spectrogram_base64 && outSpecImg) {
    outSpecImg.src = data.output_spectrogram_base64;
    outSpecImg.style.display = 'block';
    if (outCanvas) outCanvas.style.display = 'none';
    if (outPh) outPh.style.display = 'none';
  }

  // 3. Metrics HUD
  const m = data.metrics || {};
  const attVal = document.getElementById('metricAttenuation');
  if (attVal) {
    attVal.textContent = `${m.estimated_attenuation_db > 0 ? '-' : ''}${Math.abs(m.estimated_attenuation_db || 0).toFixed(1)} dB`;
  }

  const siVal = document.getElementById('metricSiSnr');
  if (siVal) {
    if (m.si_snr_improvement_db !== null && m.si_snr_improvement_db !== undefined) {
      const sign = m.si_snr_improvement_db >= 0 ? '+' : '';
      siVal.textContent = `${sign}${m.si_snr_improvement_db.toFixed(1)} dB`;
    } else {
      siVal.textContent = 'N/A (Mono)';
    }
  }

  const stoiVal = document.getElementById('metricStoi');
  if (stoiVal) {
    if (m.stoi_output !== null && m.stoi_output !== undefined) {
      stoiVal.textContent = `${m.stoi_input?.toFixed(2) || '0.00'} → ${m.stoi_output.toFixed(2)}`;
    } else {
      stoiVal.textContent = 'N/A (No Ref)';
    }
  }

  const rtVal = document.getElementById('metricRtRatio');
  if (rtVal) {
    rtVal.textContent = `${(m.realtime_ratio || 0).toFixed(2)}x`;
  }

  const rtBadge = document.getElementById('rtRatioBadge');
  if (rtBadge) {
    if (m.realtime_ratio && m.realtime_ratio < 1.0) {
      rtBadge.textContent = '< 1.0x REAL-TIME READY';
      rtBadge.className = 'pill-badge badge-green-glow';
    } else {
      rtBadge.textContent = 'BATCH MODE';
      rtBadge.className = 'pill-badge badge-red-glow';
    }
  }

  // 4. Latency Breakdown
  const cap = m.capture_ms || 0.8;
  const anc = m.anc_ms || 0.0;
  const ai = m.ai_ms || 0.0;
  const play = m.playback_ms || 1.0;
  const total = cap + anc + ai + play;

  const latTotal = document.getElementById('latencyTotalVal');
  if (latTotal) latTotal.textContent = `Total: ${total.toFixed(1)} ms / 20.0 ms`;

  const latCap = document.getElementById('valLatCap');
  if (latCap) latCap.textContent = `${cap.toFixed(1)} ms`;
  const latAnc = document.getElementById('valLatAnc');
  if (latAnc) latAnc.textContent = `${anc.toFixed(1)} ms`;
  const latAi = document.getElementById('valLatAi');
  if (latAi) latAi.textContent = `${ai.toFixed(1)} ms`;
  const latPlay = document.getElementById('valLatPlay');
  if (latPlay) latPlay.textContent = `${play.toFixed(1)} ms`;

  const denom = Math.max(total, 0.1);
  const segCap = document.getElementById('segCap');
  if (segCap) segCap.style.width = `${(cap / denom) * 100}%`;
  const segAnc = document.getElementById('segAnc');
  if (segAnc) segAnc.style.width = `${(anc / denom) * 100}%`;
  const segAi = document.getElementById('segAi');
  if (segAi) segAi.style.width = `${(ai / denom) * 100}%`;
  const segPlay = document.getElementById('segPlay');
  if (segPlay) segPlay.style.width = `${(play / denom) * 100}%`;
}

/**
 * Instant A/B Audio Switcher
 */
function toggleABPlayback() {
  const inPlayer = document.getElementById('inputAudioPlayer');
  const outPlayer = document.getElementById('outputAudioPlayer');
  const toggleBtnText = document.getElementById('abToggleText');

  if (!inPlayer || !outPlayer) return;

  if (activeABChannel === 'output') {
    const currTime = outPlayer.currentTime;
    const isPlaying = !outPlayer.paused;
    outPlayer.pause();
    inPlayer.currentTime = currTime;
    if (isPlaying) inPlayer.play();
    activeABChannel = 'input';
    if (toggleBtnText) toggleBtnText.textContent = 'Switch to Clean (Instant A/B)';
  } else {
    const currTime = inPlayer.currentTime;
    const isPlaying = !inPlayer.paused;
    inPlayer.pause();
    outPlayer.currentTime = currTime;
    if (isPlaying) outPlayer.play();
    activeABChannel = 'output';
    if (toggleBtnText) toggleBtnText.textContent = 'Switch to Disturbed (Instant A/B)';
  }
}

/**
 * Helper: Map float [0.0, 1.0] to Magma colormap [R, G, B]
 */
function getMagmaRGB(val) {
  val = Math.max(0, Math.min(1, val));
  let r, g, b;

  if (val < 0.25) {
    const t = val / 0.25;
    r = 2 + t * 45;
    g = 2 + t * 10;
    b = 8 + t * 85;
  } else if (val < 0.5) {
    const t = (val - 0.25) / 0.25;
    r = 47 + t * 105;
    g = 12 + t * 24;
    b = 93 + t * 10;
  } else if (val < 0.75) {
    const t = (val - 0.5) / 0.25;
    r = 152 + t * 75;
    g = 36 + t * 45;
    b = 103 - t * 65;
  } else if (val < 0.9) {
    const t = (val - 0.75) / 0.15;
    r = 227 + t * 23;
    g = 81 + t * 95;
    b = 38 - t * 20;
  } else {
    const t = (val - 0.9) / 0.1;
    r = 250 + t * 5;
    g = 176 + t * 78;
    b = 18 + t * 150;
  }

  return [Math.round(r), Math.round(g), Math.round(b)];
}

/**
 * Render initial rich spectrogram visuals matching Image 1 aesthetics
 */
function drawInitialSpectrograms() {
  const inCanvas = document.getElementById('inputSpecCanvas');
  const outCanvas = document.getElementById('outputSpecCanvas');
  if (!inCanvas || !outCanvas) return;

  const W = inCanvas.width;
  const H = inCanvas.height;

  // Speech utterance centers across 10 seconds (in normalized x coords [0, 1])
  const bursts = [
    { x: 0.12, w: 0.045 },
    { x: 0.20, w: 0.055 },
    { x: 0.32, w: 0.065 },
    { x: 0.42, w: 0.050 },
    { x: 0.54, w: 0.060 },
    { x: 0.67, w: 0.055 },
    { x: 0.78, w: 0.065 },
    { x: 0.89, w: 0.050 }
  ];

  // Formant relative heights (0 = bottom / 0 kHz, 1 = top / 8 kHz)
  const formants = [0.08, 0.16, 0.28, 0.42, 0.56];

  // 1. Draw Input Spectrogram (Heavy Engine Noise + Speech)
  const inCtx = inCanvas.getContext('2d');
  const inImgData = inCtx.createImageData(W, H);
  const inData = inImgData.data;

  // 2. Draw Enhanced Spectrogram (Noise-Free Clean Speech)
  const outCtx = outCanvas.getContext('2d');
  const outImgData = outCtx.createImageData(W, H);
  const outData = outImgData.data;

  for (let y = 0; y < H; y++) {
    // freq norm: 0.0 at bottom (y = H - 1), 1.0 at top (y = 0)
    const fn = (H - 1 - y) / H;

    for (let x = 0; x < W; x++) {
      const xn = x / W;
      const idx = (y * W + x) * 4;

      // Calculate speech intensity at (xn, fn)
      let speechVal = 0;
      for (let b = 0; b < bursts.length; b++) {
        const dx = (xn - bursts[b].x) / bursts[b].w;
        if (Math.abs(dx) < 2.0) {
          const envelope = Math.exp(-0.5 * dx * dx);
          // Multiple harmonic formant bands
          let formantSum = 0;
          for (let f = 0; f < formants.length; f++) {
            const df = (fn - formants[f]) / 0.035;
            formantSum += Math.exp(-0.5 * df * df);
          }
          speechVal += envelope * (formantSum * 0.55 + (fn < 0.65 ? 0.25 : 0.05));
        }
      }
      speechVal = Math.min(1.0, speechVal);

      // --- INPUT SIGNAL ---
      // Heavy low-frequency engine drone concentrated below 2 kHz (fn < 0.25)
      let engineNoise = 0;
      if (fn < 0.32) {
        const droneFalloff = Math.exp(-fn / 0.12);
        const harmonics = 0.25 * Math.sin(fn * 95) + 0.15 * Math.sin(fn * 180);
        engineNoise = 0.55 * droneFalloff + harmonics * droneFalloff;
      }
      // Subtle background noise floor across all frequencies
      const highNoise = 0.12 * Math.random();
      const inputIntensity = Math.max(0, Math.min(1, engineNoise + speechVal * 0.75 + highNoise));

      const [ir, ig, ib] = getMagmaRGB(inputIntensity);
      inData[idx] = ir;
      inData[idx + 1] = ig;
      inData[idx + 2] = ib;
      inData[idx + 3] = 255;

      // --- ENHANCED OUTPUT ---
      // Noise completely suppressed! Pure speech with high contrast
      const outIntensity = speechVal > 0.08 ? speechVal * 0.95 : 0.02 * Math.random();
      const [or, og, ob] = getMagmaRGB(outIntensity);
      outData[idx] = or;
      outData[idx + 1] = og;
      outData[idx + 2] = ob;
      outData[idx + 3] = 255;
    }
  }

  inCtx.putImageData(inImgData, 0, 0);
  outCtx.putImageData(outImgData, 0, 0);
}
