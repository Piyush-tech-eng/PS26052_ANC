/**
 * PS26052 ANC Unified Interface Frontend Controller
 * Handles audio uploads, in-browser live microphone recording,
 * REST API communications, spectrogram rendering, and audio playback.
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
  await Promise.all([loadHardwareStatus(), loadPresets()]);
}

/**
 * Fetch hardware and available models status
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
      if (modelSelect && data.models[data.recommended_model]) {
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
 * Switch Audio Input Source Tabs
 */
function switchSourceTab(type) {
  currentSourceType = type;

  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  const badge = document.getElementById('selectedSourceBadge');

  if (type === 'preset') {
    document.getElementById('tabPresetBtn').classList.add('active');
    document.getElementById('tabPresetContent').classList.add('active');
    badge.textContent = 'PRESET';
    badge.className = 'badge badge-cyan';
  } else if (type === 'upload') {
    document.getElementById('tabUploadBtn').classList.add('active');
    document.getElementById('tabUploadContent').classList.add('active');
    badge.textContent = 'FILE UPLOAD';
    badge.className = 'badge badge-cyan';
  } else if (type === 'mic') {
    document.getElementById('tabMicBtn').classList.add('active');
    document.getElementById('tabMicContent').classList.add('active');
    badge.textContent = 'LIVE MIC';
    badge.className = 'badge badge-emerald';
    setupOscilloscopeIdle();
  }
}

/**
 * Preset selection change
 */
function onPresetSelected() {
  const select = document.getElementById('presetSelect');
  const selectedId = select.value;
  const preset = availablePresets.find(p => p.id === selectedId);

  const infoText = document.getElementById('presetDetailsText');
  if (preset && infoText) {
    const refTag = preset.has_clean_reference ? 'Ground-Truth Clean Speech Reference available (SI-SNR & STOI will be evaluated).' : 'Mono Noisy Sample.';
    infoText.textContent = `Acoustic Scenario: ${preset.category.toUpperCase()} noise at ${preset.snr}. ${refTag}`;
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
      dropzone.style.borderColor = 'rgba(56, 189, 248, 0.3)';
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
 * Mode Switching (Hybrid, AI-only, Classical-only)
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
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } catch (err) {
      alert(`Could not access microphone: ${err.message}. Please allow microphone permissions.`);
      return;
    }

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    audioInputNode = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 512;

    // Buffer collection node
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

    // Start Timer
    recordTimerInterval = setInterval(() => {
      const elapsedMs = Date.now() - recordStartTime;
      const secs = Math.floor(elapsedMs / 1000);
      const dec = Math.floor((elapsedMs % 1000) / 100);
      const mm = String(Math.floor(secs / 60)).padStart(2, '0');
      const ss = String(secs % 60).padStart(2, '0');
      timer.textContent = `${mm}:${ss}.${dec}`;
    }, 100);

    // Start Waveform Visualizer
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

    // Merge Float32Array buffers and encode to standard 16-bit PCM WAV
    const merged = new Float32Array(recordedLength);
    let offset = 0;
    for (let i = 0; i < recordedBuffers.length; i++) {
      merged.set(recordedBuffers[i], offset);
      offset += recordedBuffers[i].length;
    }

    recordedAudioBase64 = encodeWAV(merged, 16000);
    timer.textContent = 'Ready to Process';

    // Automatically trigger processing pipeline
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

  // RIFF identifier
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  // fmt sub-chunk
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  // data sub-chunk
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  // Write 16-bit PCM audio samples
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

  // Payload assembly
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
      alert('Please upload an audio file first.');
      return;
    }
    payload.audio_base64 = uploadedAudioBase64;
  } else if (currentSourceType === 'mic') {
    if (!recordedAudioBase64) {
      alert('Please record a live voice sample first.');
      return;
    }
    payload.audio_base64 = recordedAudioBase64;
  }

  // Set Loading UI
  executeBtn.disabled = true;
  spinner.style.display = 'inline-block';
  icon.style.display = 'none';
  btnText.textContent = 'PROCESSING AUDIO...';
  statusBadge.textContent = 'ACTIVE / PROCESSING';
  statusBadge.className = 'pill-val status-live';

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
    btnText.textContent = 'EXECUTE NOISE CANCELLATION';
    statusBadge.textContent = 'STANDBY / READY';
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

  if (data.input_audio_base64) {
    inPlayer.src = data.input_audio_base64;
  }

  if (data.reference_audio_base64) {
    refPlayer.src = data.reference_audio_base64;
    refBox.style.display = 'flex';
  } else {
    refBox.style.display = 'none';
  }

  if (data.enhanced_audio_base64) {
    outPlayer.src = data.enhanced_audio_base64;
    dlBtn.href = data.enhanced_audio_base64;
    dlBtn.style.display = 'inline-flex';
    abToggleBtn.disabled = false;
  }

  // 2. Spectrograms
  const inSpecImg = document.getElementById('inputSpectrogramImg');
  const outSpecImg = document.getElementById('outputSpectrogramImg');
  const diffSpecImg = document.getElementById('diffSpectrogramImg');
  const inPh = document.getElementById('inputSpecPlaceholder');
  const outPh = document.getElementById('outputSpecPlaceholder');
  const diffPh = document.getElementById('diffSpecPlaceholder');

  if (data.input_spectrogram_base64) {
    inSpecImg.src = data.input_spectrogram_base64;
    inSpecImg.style.display = 'block';
    if (inPh) inPh.style.display = 'none';
  }

  if (data.output_spectrogram_base64) {
    outSpecImg.src = data.output_spectrogram_base64;
    outSpecImg.style.display = 'block';
    if (outPh) outPh.style.display = 'none';
  }

  if (data.difference_spectrogram_base64) {
    diffSpecImg.src = data.difference_spectrogram_base64;
    diffSpecImg.style.display = 'block';
    if (diffPh) diffPh.style.display = 'none';
  }

  // 3. Metrics HUD
  const m = data.metrics || {};
  document.getElementById('metricAttenuation').textContent = `${m.estimated_attenuation_db > 0 ? '-' : ''}${Math.abs(m.estimated_attenuation_db || 0).toFixed(1)} dB`;

  if (m.si_snr_improvement_db !== null && m.si_snr_improvement_db !== undefined) {
    const sign = m.si_snr_improvement_db >= 0 ? '+' : '';
    document.getElementById('metricSiSnr').textContent = `${sign}${m.si_snr_improvement_db.toFixed(1)} dB`;
  } else {
    document.getElementById('metricSiSnr').textContent = 'N/A (Mono)';
  }

  if (m.stoi_output !== null && m.stoi_output !== undefined) {
    document.getElementById('metricStoi').textContent = `${m.stoi_input?.toFixed(2) || '0.00'} → ${m.stoi_output.toFixed(2)}`;
  } else {
    document.getElementById('metricStoi').textContent = 'N/A (No Ref)';
  }

  document.getElementById('metricRtRatio').textContent = `${(m.realtime_ratio || 0).toFixed(2)}x`;

  const rtBadge = document.getElementById('rtRatioBadge');
  if (m.realtime_ratio && m.realtime_ratio < 1.0) {
    rtBadge.textContent = `${m.realtime_ratio.toFixed(2)}x REAL-TIME READY`;
    rtBadge.className = 'badge badge-emerald';
  } else {
    rtBadge.textContent = 'HIGH LOAD / BATCH';
    rtBadge.className = 'badge badge-cyan';
  }

  // 4. Latency Breakdown
  const cap = m.capture_ms || 0.8;
  const anc = m.anc_ms || 0.0;
  const ai = m.ai_ms || 0.0;
  const play = m.playback_ms || 1.0;
  const total = cap + anc + ai + play;

  document.getElementById('latencyTotalVal').textContent = `Total: ${total.toFixed(1)} ms`;
  document.getElementById('valLatCap').textContent = `${cap.toFixed(1)}ms`;
  document.getElementById('valLatAnc').textContent = `${anc.toFixed(1)}ms`;
  document.getElementById('valLatAi').textContent = `${ai.toFixed(1)}ms`;
  document.getElementById('valLatPlay').textContent = `${play.toFixed(1)}ms`;

  const denom = Math.max(total, 0.1);
  document.getElementById('segCap').style.width = `${(cap / denom) * 100}%`;
  document.getElementById('segAnc').style.width = `${(anc / denom) * 100}%`;
  document.getElementById('segAi').style.width = `${(ai / denom) * 100}%`;
  document.getElementById('segPlay').style.width = `${(play / denom) * 100}%`;
}

/**
 * Spectrogram View Switcher (Dual vs Attenuation Heatmap)
 */
function switchSpectrogramTab(view) {
  const dualView = document.getElementById('spectrogramDualView');
  const diffView = document.getElementById('spectrogramDiffView');
  const sideBtn = document.getElementById('specTabSideBtn');
  const diffBtn = document.getElementById('specTabDiffBtn');

  if (view === 'side') {
    dualView.style.display = 'grid';
    diffView.style.display = 'none';
    sideBtn.classList.add('active');
    diffBtn.classList.remove('active');
  } else {
    dualView.style.display = 'none';
    diffView.style.display = 'block';
    sideBtn.classList.remove('active');
    diffBtn.classList.add('active');
  }
}

/**
 * Instant A/B Audio Switcher
 * Seamlessly toggles playback timestamp between noisy input and enhanced output
 */
function toggleABPlayback() {
  const inPlayer = document.getElementById('inputAudioPlayer');
  const outPlayer = document.getElementById('outputAudioPlayer');
  const toggleBtnText = document.getElementById('abToggleText');

  if (activeABChannel === 'output') {
    // Switch to Noisy Input
    const currTime = outPlayer.currentTime;
    const isPlaying = !outPlayer.paused;
    outPlayer.pause();
    inPlayer.currentTime = currTime;
    if (isPlaying) inPlayer.play();
    activeABChannel = 'input';
    toggleBtnText.textContent = 'Instant A/B: Switch to Clean';
  } else {
    // Switch to Clean Output
    const currTime = inPlayer.currentTime;
    const isPlaying = !inPlayer.paused;
    inPlayer.pause();
    outPlayer.currentTime = currTime;
    if (isPlaying) outPlayer.play();
    activeABChannel = 'output';
    toggleBtnText.textContent = 'Instant A/B: Switch to Noisy';
  }
}
