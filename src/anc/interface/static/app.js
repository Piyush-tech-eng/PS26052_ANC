/**
 * PS26052 ANC Unified Interface Frontend Controller
 * Multi-page/multi-view architecture:
 * - Studio (Audio upload/preset, cascade config, A/B audio listening station)
 * - Spectral Lab (Time-frequency comparative spectrograms & attenuation heatmaps)
 * - Pi 5 Edge Hub (Live edge hardware telemetry, gauges & connection management)
 * - Benchmarks (Latency breakdown waterfall, scenario matrix, model precision tiers)
 */

// Application State
let currentView = 'studio'; // 'studio' | 'spectrograms' | 'edge' | 'benchmarks'
let currentSourceType = 'preset'; // 'preset' | 'upload' | 'mic'
let availablePresets = [];
let uploadedAudioBase64 = null;
let recordedAudioBase64 = null;
let lastProcessResult = null;
let remoteProcessingEnabled = false;

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
  initRouting();
  await Promise.all([loadHardwareStatus(), loadPresets()]);
}

/**
 * Client-Side View Router & URL Hash Synchronization
 */
function initRouting() {
  function getTargetViewFromUrl() {
    const hash = window.location.hash.replace('#', '').toLowerCase();
    const path = window.location.pathname.replace('/', '').toLowerCase();

    const candidate = hash || path;
    if (['studio', 'spectrograms', 'edge', 'benchmarks'].includes(candidate)) {
      return candidate;
    }
    return 'studio';
  }

  // Handle initial page load
  const initialView = getTargetViewFromUrl();
  switchView(initialView, false);

  // Listen for browser Back/Forward & hash changes
  window.addEventListener('hashchange', () => {
    const target = getTargetViewFromUrl();
    switchView(target, false);
  });

  window.addEventListener('popstate', () => {
    const target = getTargetViewFromUrl();
    switchView(target, false);
  });
}

/**
 * Switch Active Application View
 */
function switchView(viewName, updateHash = true) {
  if (!['studio', 'spectrograms', 'edge', 'benchmarks'].includes(viewName)) {
    viewName = 'studio';
  }
  currentView = viewName;

  // Update Navigation Link States
  document.querySelectorAll('.nav-link').forEach(btn => {
    if (btn.getAttribute('data-view') === viewName) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // Update View Panels
  const viewIdMap = {
    studio: 'viewStudio',
    spectrograms: 'viewSpectrograms',
    edge: 'viewEdge',
    benchmarks: 'viewBenchmarks',
  };

  document.querySelectorAll('.view-panel').forEach(panel => {
    panel.classList.remove('active');
  });

  const activePanel = document.getElementById(viewIdMap[viewName]);
  if (activePanel) {
    activePanel.classList.add('active');
  }

  // Update URL Hash if requested
  if (updateHash && window.location.hash !== `#${viewName}`) {
    window.location.hash = viewName;
  }

  // View-specific initialization
  if (viewName === 'spectrograms' && lastProcessResult) {
    renderSpectrograms(lastProcessResult);
  } else if (viewName === 'edge') {
    syncEdgeViewInputs();
  }
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
 * Switch Audio Input Source Tabs (Presets vs Upload vs Mic)
 */
function switchSourceTab(type) {
  currentSourceType = type;

  document.querySelectorAll('.source-tabs .tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('#viewStudio .tab-content').forEach(c => c.classList.remove('active'));

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

  // Update benchmark vs arbitrary mode badge
  updateModeIndicator(type === 'preset');
}

/**
 * Preset Selected Handler
 */
function onPresetSelected() {
  const select = document.getElementById('presetSelect');
  const details = document.getElementById('presetDetailsText');
  const selected = availablePresets.find(p => p.id === select.value);

  if (selected) {
    details.textContent = `${selected.description || 'Defence scenario audio'} | Disturbance: ${selected.noise_type || 'Acoustic'} (${selected.snr_db !== undefined ? selected.snr_db + ' dB' : 'Low SNR'})`;
  }
  updateModeIndicator(true);
}

/**
 * File Upload Handler (Drag & Drop or File Input)
 */
function handleFileSelected(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    uploadedAudioBase64 = e.target.result;
    const banner = document.getElementById('fileLoadedBanner');
    const nameEl = document.getElementById('loadedFileName');
    const sizeEl = document.getElementById('loadedFileSize');

    nameEl.textContent = file.name;
    sizeEl.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    banner.style.display = 'flex';
    updateModeIndicator(false);
  };
  reader.readAsDataURL(file);
}

// Drag & Drop event bindings
const dropzone = document.getElementById('fileDropzone');
if (dropzone) {
  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.style.borderColor = 'var(--accent-cyan)';
    }, false);
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.style.borderColor = 'rgba(56, 189, 248, 0.3)';
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) {
      document.getElementById('audioFileInput').files = files;
      handleFileSelected({ target: { files: files } });
    }
  }, false);
}

/**
 * Microphone Recording Controller
 */
async function toggleMicRecording() {
  const btn = document.getElementById('micRecordBtn');
  const btnText = document.getElementById('micRecordBtnText');
  const timer = document.getElementById('recordTimer');

  if (!isRecording) {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioInputNode = audioContext.createMediaStreamSource(mediaStream);
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 512;

      audioInputNode.connect(analyserNode);

      // ScriptProcessorNode for raw PCM extraction
      recorderNode = audioContext.createScriptProcessor(4096, 1, 1);
      recordedBuffers = [];
      recordedLength = 0;

      recorderNode.onaudioprocess = (e) => {
        if (!isRecording) return;
        const channelData = e.inputBuffer.getChannelData(0);
        recordedBuffers.push(new Float32Array(channelData));
        recordedLength += channelData.length;
      };

      audioInputNode.connect(recorderNode);
      recorderNode.connect(audioContext.destination);

      isRecording = true;
      recordStartTime = performance.now();
      btnText.textContent = 'Stop Recording';
      btn.style.background = 'rgba(239, 68, 68, 0.35)';

      recordTimerInterval = setInterval(() => {
        const elapsed = (performance.now() - recordStartTime) / 1000;
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = (elapsed % 60).toFixed(1).padStart(4, '0');
        timer.textContent = `${mins}:${secs}`;
      }, 100);

      drawOscilloscopeActive();

    } catch (err) {
      alert(`Microphone access error: ${err.message}`);
      console.error(err);
    }
  } else {
    // Stop recording
    isRecording = false;
    clearInterval(recordTimerInterval);
    btnText.textContent = 'Record Again';
    btn.style.background = 'rgba(239, 68, 68, 0.15)';

    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    if (recorderNode) recorderNode.disconnect();
    if (audioInputNode) audioInputNode.disconnect();
    if (audioContext && audioContext.state !== 'closed') audioContext.close();

    // Flatten buffers
    const merged = new Float32Array(recordedLength);
    let offset = 0;
    for (const b of recordedBuffers) {
      merged.set(b, offset);
      offset += b.length;
    }

    recordedAudioBase64 = encodeWAV(merged, 16000);
    updateModeIndicator(false);
    setupOscilloscopeIdle();
  }
}

/**
 * Oscilloscope Canvas Renderers
 */
function setupOscilloscopeIdle() {
  const canvas = document.getElementById('micOscilloscopeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#050810';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
}

function drawOscilloscopeActive() {
  if (!isRecording) return;
  const canvas = document.getElementById('micOscilloscopeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const buffer = new Uint8Array(analyserNode.frequencyBinCount);
  analyserNode.getByteTimeDomainData(buffer);

  ctx.fillStyle = '#050810';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 2;
  ctx.strokeStyle = '#38bdf8';
  ctx.beginPath();

  const sliceWidth = canvas.width / buffer.length;
  let x = 0;

  for (let i = 0; i < buffer.length; i++) {
    const v = buffer[i] / 128.0;
    const y = (v * canvas.height) / 2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    x += sliceWidth;
  }

  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();

  animFrameId = requestAnimationFrame(drawOscilloscopeActive);
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
 * Mode Change Handler (Cascade selector: Hybrid vs AI-only vs Classical)
 */
function onModeChanged() {
  const mode = document.getElementById('pipelineModeSelect').value;
  const modelGroup = document.getElementById('modelSelectGroup');
  const tapsGroup = document.getElementById('filterTapsGroup');
  const stepGroup = document.getElementById('stepSizeGroup');

  if (mode === 'hybrid') {
    modelGroup.style.display = 'flex';
    tapsGroup.style.display = 'flex';
    stepGroup.style.display = 'flex';
  } else if (mode === 'ai_only') {
    modelGroup.style.display = 'flex';
    tapsGroup.style.display = 'none';
    stepGroup.style.display = 'none';
  } else if (mode === 'anc_only') {
    modelGroup.style.display = 'none';
    tapsGroup.style.display = 'flex';
    stepGroup.style.display = 'flex';
  }
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
    const apiEndpoint = remoteProcessingEnabled ? '/api/process-remote' : '/api/process';

    if (remoteProcessingEnabled) {
      payload.pi_host = document.getElementById('piHostInput')?.value || 'raspberrypi.local';
      payload.pi_port = parseInt(document.getElementById('piPortInput')?.value || '8090', 10);
    }

    const res = await fetch(apiEndpoint, {
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

    // Show ready badge on Spectral Lab tab
    const specBadge = document.getElementById('specReadyBadge');
    if (specBadge) specBadge.style.display = 'inline-block';

    // Update Pi telemetry if returned
    if (data.telemetry) {
      renderPiTelemetry(data.telemetry, data.metrics);
    }

    updateModeIndicator(data.has_clean_reference);

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
 * Render Audio, Spectrograms, Scorecard & Telemetry
 */
function renderResults(data) {
  // 1. Audio Players in Studio View
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

  // 2. Scorecard KPIs
  const m = data.metrics || {};
  const attenEl = document.getElementById('metricAttenuation');
  if (attenEl) {
    attenEl.textContent = `${m.estimated_attenuation_db > 0 ? '-' : ''}${Math.abs(m.estimated_attenuation_db || 0).toFixed(1)} dB`;
  }

  const siSnrEl = document.getElementById('metricSiSnr');
  if (siSnrEl) {
    if (m.si_snr_improvement_db !== null && m.si_snr_improvement_db !== undefined) {
      const sign = m.si_snr_improvement_db >= 0 ? '+' : '';
      siSnrEl.textContent = `${sign}${m.si_snr_improvement_db.toFixed(1)} dB`;
    } else {
      siSnrEl.textContent = 'N/A (No Ref)';
    }
  }

  const stoiEl = document.getElementById('metricStoi');
  if (stoiEl) {
    if (m.stoi_output !== null && m.stoi_output !== undefined) {
      stoiEl.textContent = `${m.stoi_input?.toFixed(2) || '0.00'} → ${m.stoi_output.toFixed(2)}`;
    } else {
      stoiEl.textContent = 'N/A (No Ref)';
    }
  }

  const rtEl = document.getElementById('metricRtRatio');
  if (rtEl) {
    rtEl.textContent = `${(m.realtime_ratio || 0).toFixed(2)}x`;
  }

  const rtBadge = document.getElementById('rtRatioBadge');
  if (rtBadge) {
    if (m.realtime_ratio && m.realtime_ratio < 1.0) {
      rtBadge.textContent = `${m.realtime_ratio.toFixed(2)}x REAL-TIME READY`;
      rtBadge.className = 'badge badge-emerald';
    } else {
      rtBadge.textContent = 'HIGH LOAD / BATCH';
      rtBadge.className = 'badge badge-cyan';
    }
  }

  // 3. Render Spectrograms in Spectral Lab View
  renderSpectrograms(data);

  // 4. Update Latency Waterfall in Benchmarks View
  updateLatencyWaterfall(m);
}

/**
 * Render High-Resolution Spectrograms in Spectral Lab
 */
function renderSpectrograms(data) {
  const inSpecImg = document.getElementById('inputSpectrogramImg');
  const outSpecImg = document.getElementById('enhancedSpectrogramImg');
  const diffSpecImg = document.getElementById('diffSpectrogramImg');
  const refSpecImg = document.getElementById('refSpectrogramImg');
  const refItem = document.getElementById('refSpectrogramItem');
  const placeholder = document.getElementById('specPlaceholder');
  const sideWrapper = document.getElementById('specSideViews');

  if (placeholder) placeholder.style.display = 'none';
  if (sideWrapper) sideWrapper.style.display = 'flex';

  if (data.input_spectrogram_base64 && inSpecImg) {
    inSpecImg.src = data.input_spectrogram_base64;
  }

  if (data.output_spectrogram_base64 && outSpecImg) {
    outSpecImg.src = data.output_spectrogram_base64;
  }

  if (data.difference_spectrogram_base64 && diffSpecImg) {
    diffSpecImg.src = data.difference_spectrogram_base64;
  }

  if (data.reference_audio_base64 && refItem && refSpecImg) {
    refItem.style.display = 'flex';
    // If reference spectrogram isn't separate, use input/output context
    refSpecImg.src = data.input_spectrogram_base64;
  } else if (refItem) {
    refItem.style.display = 'none';
  }
}

/**
 * Update Latency Waterfall in Benchmarks View
 */
function updateLatencyWaterfall(m) {
  const cap = m.capture_ms || 0.8;
  const anc = m.anc_ms || 0.0;
  const ai = m.ai_ms || 5.7;
  const play = m.playback_ms || 1.0;
  const total = cap + anc + ai + play;

  const totalEl = document.getElementById('latencyTotalVal');
  if (totalEl) totalEl.textContent = `Total: ${total.toFixed(1)} ms`;

  const valCap = document.getElementById('valLatCap');
  const valAnc = document.getElementById('valLatAnc');
  const valAi = document.getElementById('valLatAi');
  const valPlay = document.getElementById('valLatPlay');

  if (valCap) valCap.textContent = `${cap.toFixed(1)} ms`;
  if (valAnc) valAnc.textContent = `${anc.toFixed(1)} ms`;
  if (valAi) valAi.textContent = `${ai.toFixed(1)} ms`;
  if (valPlay) valPlay.textContent = `${play.toFixed(1)} ms`;

  const denom = Math.max(total, 0.1);
  const segCap = document.getElementById('segCap');
  const segAnc = document.getElementById('segAnc');
  const segAi = document.getElementById('segAi');
  const segPlay = document.getElementById('segPlay');

  if (segCap) segCap.style.width = `${(cap / denom) * 100}%`;
  if (segAnc) segAnc.style.width = `${(anc / denom) * 100}%`;
  if (segAi) segAi.style.width = `${(ai / denom) * 100}%`;
  if (segPlay) segPlay.style.width = `${(play / denom) * 100}%`;
}

/**
 * Spectrogram View Switcher in Spectral Lab (Side vs Diff)
 */
function switchSpectrogramTab(view) {
  const sideWrapper = document.getElementById('specSideViews');
  const diffWrapper = document.getElementById('specDiffView');
  const sideBtn = document.getElementById('specTabSideBtn');
  const diffBtn = document.getElementById('specTabDiffBtn');
  const placeholder = document.getElementById('specPlaceholder');

  // If no audio processed yet, stay on placeholder
  if (placeholder && placeholder.style.display !== 'none') return;

  if (view === 'side') {
    if (sideWrapper) sideWrapper.style.display = 'flex';
    if (diffWrapper) diffWrapper.style.display = 'none';
    if (sideBtn) sideBtn.classList.add('active');
    if (diffBtn) diffBtn.classList.remove('active');
  } else {
    if (sideWrapper) sideWrapper.style.display = 'none';
    if (diffWrapper) diffWrapper.style.display = 'block';
    if (sideBtn) sideBtn.classList.remove('active');
    if (diffBtn) diffBtn.classList.add('active');
  }
}

/**
 * Instant A/B Audio Switcher (Channel A Noisy vs Channel C Clean)
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
    if (toggleBtnText) toggleBtnText.textContent = 'Instant A/B: Switch to Clean';
  } else {
    const currTime = inPlayer.currentTime;
    const isPlaying = !inPlayer.paused;
    inPlayer.pause();
    outPlayer.currentTime = currTime;
    if (isPlaying) outPlayer.play();
    activeABChannel = 'output';
    if (toggleBtnText) toggleBtnText.textContent = 'Instant A/B: Switch to Noisy';
  }
}

/**
 * Pi 5 Remote Processing Toggle in Studio
 */
function onPiToggleChanged() {
  const toggle = document.getElementById('piRemoteToggle');
  remoteProcessingEnabled = toggle ? toggle.checked : false;

  const modeBadge = document.getElementById('piModeBadge');
  const quickInfo = document.getElementById('piQuickInfo');
  const hostDisplay = document.getElementById('piHostDisplay');
  const hostInput = document.getElementById('piHostInput');
  const portInput = document.getElementById('piPortInput');

  if (remoteProcessingEnabled) {
    if (modeBadge) modeBadge.style.display = 'inline-block';
    if (quickInfo) quickInfo.style.display = 'flex';
    const host = hostInput ? hostInput.value : 'raspberrypi.local';
    const port = portInput ? portInput.value : '8090';
    if (hostDisplay) hostDisplay.textContent = `${host}:${port}`;
  } else {
    if (modeBadge) modeBadge.style.display = 'none';
    if (quickInfo) quickInfo.style.display = 'none';
  }
}

function syncEdgeViewInputs() {
  const hostDisplay = document.getElementById('piHostDisplay');
  const hostInput = document.getElementById('piHostInput');
  const portInput = document.getElementById('piPortInput');

  if (hostDisplay && hostInput && portInput) {
    hostDisplay.textContent = `${hostInput.value || 'raspberrypi.local'}:${portInput.value || '8090'}`;
  }
}

/**
 * Check Pi 5 Health & Connectivity (via /api/pi-health server proxy)
 */
async function checkPiHealth() {
  const piHost = document.getElementById('piHostInput')?.value || 'raspberrypi.local';
  const piPort = document.getElementById('piPortInput')?.value || '8090';
  const dot = document.getElementById('piHealthDot');
  const text = document.getElementById('piHealthText');
  const banner = document.getElementById('piStatusBanner');
  const msg = document.getElementById('piStatusMsg');
  const edgeBadge = document.getElementById('piEdgeStatusBadge');
  const navDot = document.getElementById('navPiDot');

  if (dot) dot.className = 'health-dot health-checking';
  if (text) text.textContent = 'Testing...';
  if (msg) msg.textContent = `Connecting to ${piHost}:${piPort}...`;

  try {
    const proxyUrl = `/api/pi-health?host=${encodeURIComponent(piHost)}&port=${encodeURIComponent(piPort)}`;
    const res = await fetch(proxyUrl);
    const data = await res.json();

    if (data.connected && data.data && data.data.status === 'ok') {
      if (dot) dot.className = 'health-dot health-ok';
      if (text) text.textContent = 'Connected';
      if (edgeBadge) {
        edgeBadge.textContent = 'ONLINE';
        edgeBadge.className = 'badge badge-emerald';
      }
      if (navDot) navDot.className = 'nav-dot connected';
      if (msg) {
        msg.textContent = `Successfully connected to Pi 5 (${data.data.platform || 'ARM Linux'}). Model: ${data.data.model_name || 'DTLN loaded'}.`;
      }
      syncEdgeViewInputs();
    } else {
      if (dot) dot.className = 'health-dot health-error';
      if (text) text.textContent = 'Unreachable';
      if (edgeBadge) {
        edgeBadge.textContent = 'OFFLINE';
        edgeBadge.className = 'badge badge-amber';
      }
      if (navDot) navDot.className = 'nav-dot';
      if (msg) {
        msg.textContent = data.error || `Unable to reach Pi 5 at ${piHost}:${piPort}. Ensure inference_server.py is running.`;
      }
    }
  } catch (err) {
    if (dot) dot.className = 'health-dot health-error';
    if (text) text.textContent = 'Error';
    if (edgeBadge) {
      edgeBadge.textContent = 'ERROR';
      edgeBadge.className = 'badge badge-amber';
    }
    if (navDot) navDot.className = 'nav-dot';
    if (msg) {
      msg.textContent = `Connection error: ${err.message}`;
    }
  }
}

/**
 * Render Pi 5 Edge Telemetry Gauges in Edge Hub View
 */
function renderPiTelemetry(telemetry, metrics) {
  // 1. CPU Load Gauge
  const cpuVal = telemetry.cpu_load_percent || 0;
  const cpuEl = document.getElementById('piMetricCpu');
  const cpuBar = document.getElementById('piBarCpu');
  if (cpuEl) cpuEl.textContent = `${cpuVal.toFixed(1)}%`;
  if (cpuBar) cpuBar.style.width = `${Math.min(cpuVal, 100)}%`;

  // 2. RAM Memory Gauge
  const ramUsed = telemetry.ram_used_mb || 0;
  const ramTotal = telemetry.ram_total_mb || 0;
  const ramPct = telemetry.ram_percent || 0;
  const ramEl = document.getElementById('piMetricRam');
  const ramPctEl = document.getElementById('piMetricRamPct');
  const ramBar = document.getElementById('piBarRam');
  if (ramEl) ramEl.textContent = `${ramUsed.toFixed(0)} / ${ramTotal.toFixed(0)} MB`;
  if (ramPctEl) ramPctEl.textContent = `${ramPct.toFixed(0)}%`;
  if (ramBar) ramBar.style.width = `${Math.min(ramPct, 100)}%`;

  // 3. SoC Temperature Gauge
  const temp = telemetry.temperature_c;
  const tempEl = document.getElementById('piMetricTemp');
  const tempBar = document.getElementById('piBarTemp');
  const tempStatus = document.getElementById('piTempStatus');

  if (temp !== null && temp !== undefined && tempEl) {
    tempEl.textContent = `${temp.toFixed(1)}°C`;
    if (tempBar) tempBar.style.width = `${Math.min((temp / 85) * 100, 100)}%`;
    if (tempStatus) {
      if (temp < 60) {
        tempStatus.textContent = 'COOL (<60°C)';
        tempStatus.className = 'gauge-badge badge-emerald';
      } else if (temp < 75) {
        tempStatus.textContent = 'WARM';
        tempStatus.className = 'gauge-badge badge-amber';
      } else {
        tempStatus.textContent = 'THROTTLING';
        tempStatus.className = 'gauge-badge';
        tempStatus.style.background = 'rgba(239, 68, 68, 0.2)';
        tempStatus.style.color = '#ef4444';
      }
    }
  } else if (tempEl) {
    tempEl.textContent = 'N/A';
  }

  // 4. Inference Time Gauge
  const timeEl = document.getElementById('piMetricTime');
  const rtEl = document.getElementById('piMetricRtRatio');
  const rtBar = document.getElementById('piBarRt');
  const budgetStatus = document.getElementById('piBudgetStatus');

  if (metrics) {
    const aiMs = metrics.ai_ms || 0;
    const rtRatio = metrics.realtime_ratio || 0;
    if (timeEl) timeEl.textContent = `${aiMs.toFixed(1)} ms`;
    if (rtEl) rtEl.textContent = `${rtRatio.toFixed(3)}x real-time ratio`;
    if (rtBar) rtBar.style.width = `${Math.min((aiMs / 20) * 100, 100)}%`;
    if (budgetStatus) {
      if (aiMs <= 20) {
        budgetStatus.textContent = '< 20ms BUDGET OK';
        budgetStatus.className = 'gauge-badge badge-emerald';
      } else {
        budgetStatus.textContent = 'EXCEEDS BUDGET';
        budgetStatus.className = 'gauge-badge badge-amber';
      }
    }
  }
}

/**
 * Update Mode Indicator Badge (Benchmark vs Arbitrary Upload)
 */
function updateModeIndicator(hasCleanReference) {
  const indicator = document.getElementById('modeIndicator');
  const benchBadge = document.getElementById('modeBenchmarkBadge');
  const arbBadge = document.getElementById('modeArbitraryBadge');

  if (!indicator) return;
  indicator.style.display = 'block';

  if (hasCleanReference) {
    if (benchBadge) benchBadge.style.display = 'inline-flex';
    if (arbBadge) arbBadge.style.display = 'none';
  } else {
    if (benchBadge) benchBadge.style.display = 'none';
    if (arbBadge) arbBadge.style.display = 'inline-flex';
  }
}
