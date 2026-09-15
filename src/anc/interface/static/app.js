/**
 * AI-Powered ANC Dashboard — Frontend Application
 * Handles view routing, API communication, chart rendering, and all UI interactions.
 */

/* ── STATE ── */
let currentView = 'upload';
let currentSource = 'preset';
let selectedPresetId = null;
let uploadedFileB64 = null;
let uploadedFileName = null;
let micMediaRecorder = null;
let micAudioChunks = [];
let micRecordingB64 = null;
let micTimerInterval = null;
let isShowingEnhanced = true;
let lastResult = null;
let chartInstances = {};

/* ── INIT ── */
document.addEventListener('DOMContentLoaded', () => {
  fetchPresets();
  setupDropzone();
  setSystemStatus('idle');
  // Restore theme preference
  const saved = localStorage.getItem('ancTheme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (saved === 'dark' || (!saved && prefersDark)) {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
});

/* ── DARK MODE ── */
function toggleDarkMode() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  localStorage.setItem('ancTheme', isDark ? 'light' : 'dark');
}

/* ── UI HELPERS ── */
function setSystemStatus(state) {
  const badge = document.getElementById('systemStatusBadge');
  if (!badge) return;
  
  badge.className = `header-badge header-badge--${state}`;
  
  switch(state) {
    case 'idle': badge.textContent = 'System Ready'; break;
    case 'uploading': badge.textContent = 'Uploading...'; break;
    case 'processing': badge.textContent = 'Processing Signal...'; break;
    case 'complete': badge.textContent = 'Enhancement Complete'; break;
  }
}

/* ── VIEW ROUTING ── */
function showView(name) {
  ['upload', 'processing', 'results', 'analysis'].forEach(v => {
    const el = document.getElementById(`view${v.charAt(0).toUpperCase() + v.slice(1)}`);
    if (el) el.classList.toggle('active', v === name);
  });

  ['Upload', 'Results', 'Analysis'].forEach(n => {
    const btn = document.getElementById(`nav${n}`);
    const badge = document.getElementById(`navStep${n}`);
    const isCurrent = (n.toLowerCase() === name);
    
    if (btn) {
      btn.classList.toggle('active', isCurrent);
      btn.setAttribute('aria-current', isCurrent ? 'page' : 'false');
    }
    
    if (badge) {
      badge.className = 'nav-step-badge';
      if (isCurrent) {
        badge.classList.add('nav-step-badge--active');
        badge.textContent = '●';
      } else if ((name === 'results' || name === 'analysis') && n === 'Upload' || (name === 'analysis' && n === 'Results')) {
        badge.classList.add('nav-step-badge--done');
        badge.textContent = '✓';
      } else {
        badge.classList.add('nav-step-badge--pending');
        badge.textContent = '○';
      }
    }
  });

  currentView = name;

  // Resize charts when switching to analysis
  if (name === 'analysis') {
    setTimeout(() => {
      Object.values(chartInstances).forEach(c => { if (c) c.resize(); });
    }, 100);
  }

  // Show/hide latency badge based on source mode
  const latencyBadge = document.getElementById('latencyBadge');
  if (latencyBadge) {
    latencyBadge.style.display = (currentSource === 'mic') ? 'inline-flex' : 'none';
  }
}

/* ── SOURCE TABS ── */
function switchSource(tab) {
  currentSource = tab;
  ['preset', 'upload', 'mic'].forEach(t => {
    const btn = document.getElementById(`tab${t.charAt(0).toUpperCase() + t.slice(1)}`);
    const panel = document.getElementById(`panel${t.charAt(0).toUpperCase() + t.slice(1)}`);
    if (btn) {
      btn.classList.toggle('active', t === tab);
      btn.setAttribute('aria-selected', t === tab ? 'true' : 'false');
    }
    if (panel) panel.classList.toggle('active', t === tab);
  });

  // Show latency badge only in mic mode
  const latencyBadge = document.getElementById('latencyBadge');
  if (latencyBadge) {
    latencyBadge.style.display = (tab === 'mic') ? 'inline-flex' : 'none';
  }
}

/* ── PRESETS ── */
async function fetchPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    const select = document.getElementById('presetSelect');
    if (!select || !data.presets) return;

    select.innerHTML = '<option value="" disabled selected>Select a defence scenario…</option>';
    data.presets.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.name;
      opt.dataset.hasRef = p.has_clean_reference ? 'true' : 'false';
      select.appendChild(opt);
    });

    if (data.presets.length > 0) {
      select.value = data.presets[0].id;
      onPresetSelected();
    }
  } catch (err) {
    console.error('Failed to fetch presets:', err);
  }
}

function onPresetSelected() {
  const select = document.getElementById('presetSelect');
  selectedPresetId = select ? select.value : null;
  const details = document.getElementById('presetDetailsText');
  if (details && selectedPresetId) {
    const opt = select.options[select.selectedIndex];
    const hasRef = opt?.dataset?.hasRef === 'true';
    details.textContent = `Preset '${selectedPresetId}' selected. ${hasRef ? 'Clean reference available — real SNR/STOI/PESQ will be computed.' : 'No clean reference — estimation metrics only.'}`;
  }

  // Show file details card
  const card = document.getElementById('fileDetailsCard');
  if (card && selectedPresetId) {
    card.classList.add('visible');
    document.getElementById('detailFilename').textContent = selectedPresetId + '_noisy.wav';
    document.getElementById('detailFormat').textContent = 'WAV';
    document.getElementById('detailSampleRate').textContent = '16 kHz';
    document.getElementById('detailChannels').textContent = '1';
    document.getElementById('detailDuration').textContent = '—';
    document.getElementById('detailSize').textContent = '—';
  }
}

/* ── FILE UPLOAD ── */
function handleFileSelected(event) {
  const file = event.target.files[0];
  if (!file) return;

  uploadedFileName = file.name;
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadedFileB64 = e.target.result;
    const banner = document.getElementById('fileBanner');
    if (banner) {
      banner.style.display = 'flex';
      document.getElementById('loadedFileName').textContent = file.name;
      document.getElementById('loadedFileSize').textContent = `${(file.size / 1024).toFixed(1)} KB`;
    }
    
    setSystemStatus('uploading');
    setTimeout(() => setSystemStatus('idle'), 500);

    // Show file details
    const card = document.getElementById('fileDetailsCard');
    if (card) {
      card.classList.add('visible');
      document.getElementById('detailFilename').textContent = file.name;
      document.getElementById('detailFormat').textContent = file.name.split('.').pop().toUpperCase();
      document.getElementById('detailSize').textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
      document.getElementById('detailDuration').textContent = '—';
      document.getElementById('detailSampleRate').textContent = '16 kHz';
      document.getElementById('detailChannels').textContent = '1';
    }
  };
  reader.readAsDataURL(file);
}

function clearFileSelection() {
  uploadedFileB64 = null;
  uploadedFileName = null;
  selectedPresetId = null;
  document.getElementById('fileDetailsCard').classList.remove('visible');
  document.getElementById('fileBanner').style.display = 'none';
  document.getElementById('audioFileInput').value = '';
}

function setupDropzone() {
  const dz = document.getElementById('fileDropzone');
  if (!dz) return;

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
    dz.addEventListener(evt, e => e.preventDefault(), false);
  });

  dz.addEventListener('dragenter', () => dz.classList.add('drag-over'));
  dz.addEventListener('dragover', () => dz.classList.add('drag-over'));
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    dz.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      document.getElementById('audioFileInput').files = files;
      handleFileSelected({ target: { files } });
    }
  });
}

/* ── MIC RECORDING ── */
async function toggleMicRecording() {
  const btn = document.getElementById('micRecordBtn');
  const btnText = document.getElementById('micRecordBtnText');
  const timer = document.getElementById('recordTimer');

  if (micMediaRecorder && micMediaRecorder.state === 'recording') {
    micMediaRecorder.stop();
    if (btnText) btnText.textContent = 'Start Recording';
    clearInterval(micTimerInterval);
  } else {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micMediaRecorder = new MediaRecorder(stream);
      micAudioChunks = [];

      micMediaRecorder.ondataavailable = e => micAudioChunks.push(e.data);
      micMediaRecorder.onstop = async () => {
        const blob = new Blob(micAudioChunks, { type: 'audio/wav' });
        const reader = new FileReader();
        reader.onload = e => {
          micRecordingB64 = e.target.result;
          if (timer) timer.textContent = 'REC DONE';
        };
        reader.readAsDataURL(blob);
      };

      micMediaRecorder.start();
      if (btnText) btnText.textContent = 'Stop Recording';

      let seconds = 0;
      micTimerInterval = setInterval(() => {
        seconds += 0.1;
        if (timer) timer.textContent = seconds.toFixed(1) + 's';
      }, 100);
    } catch (err) {
      alert('Microphone access denied or unavailable: ' + err.message);
    }
  }
}

/* ── ADVANCED CONFIG ── */
function toggleAdvanced() {
  const toggle = document.getElementById('advancedToggle');
  const panel = document.getElementById('advancedPanel');
  const isOpen = toggle.classList.toggle('open');
  panel.classList.toggle('open', isOpen);
  toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
}

function onModeChanged() {
  const mode = document.getElementById('pipelineModeSelect').value;
  const filterGroup = document.getElementById('filterTapsGroup');
  const stepGroup = document.getElementById('stepSizeGroup');
  const modelGroup = document.getElementById('modelSelectGroup');

  if (filterGroup) filterGroup.style.display = (mode === 'ai_only') ? 'none' : '';
  if (stepGroup) stepGroup.style.display = (mode === 'ai_only') ? 'none' : '';
  if (modelGroup) modelGroup.style.display = (mode === 'anc_only') ? 'none' : '';
}

/* ── PLAYER TABS (A/B toggle) ── */
function switchPlayerTab(tab) {
  isShowingEnhanced = (tab === 'enhanced');
  const enhancedPlayer = document.getElementById('enhancedPlayer');
  const originalPlayer = document.getElementById('originalPlayer');
  const tabEnhanced = document.getElementById('playerTabEnhanced');
  const tabOriginal = document.getElementById('playerTabOriginal');

  if (tabEnhanced) {
    tabEnhanced.classList.toggle('active', isShowingEnhanced);
    tabEnhanced.setAttribute('aria-selected', isShowingEnhanced ? 'true' : 'false');
  }
  if (tabOriginal) {
    tabOriginal.classList.toggle('active', !isShowingEnhanced);
    tabOriginal.setAttribute('aria-selected', !isShowingEnhanced ? 'true' : 'false');
  }

  if (enhancedPlayer && originalPlayer) {
    // Sync playback position for instant A/B comparison
    if (isShowingEnhanced) {
      enhancedPlayer.style.display = '';
      originalPlayer.style.display = 'none';
      if (!originalPlayer.paused) {
        enhancedPlayer.currentTime = originalPlayer.currentTime;
        originalPlayer.pause();
        enhancedPlayer.play();
      }
    } else {
      enhancedPlayer.style.display = 'none';
      originalPlayer.style.display = '';
      if (!enhancedPlayer.paused) {
        originalPlayer.currentTime = enhancedPlayer.currentTime;
        enhancedPlayer.pause();
        originalPlayer.play();
      }
    }
  }
}

/* ── ANALYSIS TABS ── */
function switchAnalysisTab(tab) {
  ['overview', 'mod3', 'mod4', 'mod5'].forEach(t => {
    const btn = document.getElementById(`aTab${t.charAt(0).toUpperCase() + t.slice(1)}`);
    const pane = document.getElementById(`aPane${t.charAt(0).toUpperCase() + t.slice(1)}`);
    if (btn) {
      btn.classList.toggle('active', t === tab);
      btn.setAttribute('aria-selected', t === tab ? 'true' : 'false');
    }
    if (pane) pane.classList.toggle('active', t === tab);
  });

  setTimeout(() => {
    Object.values(chartInstances).forEach(c => { if (c) c.resize(); });
  }, 50);
}

/* ── PROCESSING STEPPER ANIMATION ── */
function animateStepper(stepIndex, status) {
  for (let i = 0; i < 8; i++) {
    const step = document.getElementById(`step${i}`);
    const statusEl = document.getElementById(`step${i}Status`);
    if (!step) continue;

    step.classList.remove('completed', 'active', 'pending');
    if (i < stepIndex) {
      step.classList.add('completed');
      if (statusEl) statusEl.textContent = 'Completed';
    } else if (i === stepIndex) {
      step.classList.add('active');
      if (statusEl) statusEl.textContent = status || 'In Progress';
    } else {
      step.classList.add('pending');
      if (statusEl) statusEl.textContent = 'Pending';
    }
  }
}

function updateProgress(percent) {
  const fill = document.getElementById('progressFill');
  const text = document.getElementById('progressPercent');
  if (fill) fill.style.width = `${percent}%`;
  if (text) text.textContent = `${Math.round(percent)}%`;
}

/* ── MAIN PIPELINE EXECUTION ── */
async function runProcessingPipeline() {
  const btn = document.getElementById('executeBtn');
  const spinner = document.getElementById('executeSpinner');
  const icon = document.getElementById('executeIcon');
  const btnText = document.getElementById('executeBtnText');

  // Build payload
  const mode = document.getElementById('pipelineModeSelect').value;
  const modelName = document.getElementById('modelSelect').value;
  const filterLength = parseInt(document.getElementById('filterTapsSelect').value, 10);
  const stepSize = parseFloat(document.getElementById('stepSizeSelect').value);

  const payload = { mode, model_name: modelName, filter_length: filterLength, step_size: stepSize };

  if (currentSource === 'preset') {
    if (!selectedPresetId) { alert('Please select a preset audio scenario.'); return; }
    payload.preset_id = selectedPresetId;
  } else if (currentSource === 'upload') {
    if (!uploadedFileB64) { alert('Please upload an audio file first.'); return; }
    payload.audio_base64 = uploadedFileB64;
  } else if (currentSource === 'mic') {
    if (!micRecordingB64) { alert('Please record audio using the microphone first.'); return; }
    payload.audio_base64 = micRecordingB64;
  }

  // Switch to processing view
  showView('processing');

  // Set pipeline details dynamically
  const pipelineNames = { hybrid: 'Hybrid (FxNLMS + Deep AI)', ai_only: 'AI-Only (Neural DTLN)', anc_only: 'Classical ANC-Only' };
  document.getElementById('pipelineUsedText').textContent = pipelineNames[mode] || mode;
  document.getElementById('aiModelStatusText').textContent = `Loading ${modelName}…`;
  document.getElementById('pipelineStatusText').textContent = 'Processing…';
  document.getElementById('currentStageText').textContent = 'Initializing pipeline…';

  // Update AI labels in flow diagram
  if (mode === 'anc_only') {
    document.getElementById('flowAiLabel').textContent = 'AI';
    document.getElementById('flowAiSublabel').textContent = '(skipped)';
    document.getElementById('step4Label').textContent = 'AI Enhancement (Skipped)';
  } else {
    document.getElementById('flowAiLabel').textContent = 'AI';
    document.getElementById('flowAiSublabel').textContent = modelName;
    document.getElementById('step4Label').textContent = 'AI Enhancement';
  }

  // Disable button
  if (btn) btn.disabled = true;
  if (spinner) spinner.style.display = 'inline-block';
  if (icon) icon.style.display = 'none';
  if (btnText) btnText.textContent = 'Processing…';
  
  setSystemStatus('processing');

  // Animate stepper progressively
  const stepNames = ['Validating input', 'Analyzing signal', 'Characterizing noise', 'Applying adaptive filter',
    'Running AI enhancement', 'Computing metrics', 'Generating visualizations', 'Exporting output'];

  let stepIdx = 0;
  const stepperTimer = setInterval(() => {
    if (stepIdx < 4) {
      animateStepper(stepIdx, 'In Progress');
      updateProgress((stepIdx / 8) * 100);
      document.getElementById('currentStageText').textContent = stepNames[stepIdx] + '…';
      stepIdx++;
    }
  }, 600);

  try {
    const res = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    clearInterval(stepperTimer);

    const result = await res.json();
    if (!result.success) {
      alert('Processing Error: ' + (result.error || result.message));
      resetExecuteBtn();
      showView('upload');
      return;
    }

    // Complete remaining steps quickly
    for (let i = stepIdx; i <= 7; i++) {
      animateStepper(i, i < 7 ? 'Completed' : 'In Progress');
      updateProgress(((i + 1) / 8) * 100);
      document.getElementById('currentStageText').textContent = stepNames[i] + '…';
      await new Promise(r => setTimeout(r, 200));
    }
    animateStepper(8, ''); // all completed
    updateProgress(100);

    // Update AI model status with real name from backend
    document.getElementById('aiModelStatusText').textContent = result.model_name || modelName;
    document.getElementById('pipelineStatusText').textContent = 'Completed successfully';
    document.getElementById('currentStageText').textContent = 'Processing complete — switching to results…';

    lastResult = result;

    // Brief pause then switch to results
    await new Promise(r => setTimeout(r, 800));
    renderResults(result);
    showView('results');

  } catch (err) {
    clearInterval(stepperTimer);
    alert('Server communication failed: ' + err.message);
    showView('upload');
  } finally {
    resetExecuteBtn();
  }
}

function resetExecuteBtn() {
  const btn = document.getElementById('executeBtn');
  const spinner = document.getElementById('executeSpinner');
  const icon = document.getElementById('executeIcon');
  const btnText = document.getElementById('executeBtnText');

  if (btn) btn.disabled = false;
  if (spinner) spinner.style.display = 'none';
  if (icon) icon.style.display = '';
  if (btnText) btnText.textContent = 'Analyze & Enhance Audio';
}

/* ── RENDER RESULTS ── */
function renderResults(res) {
  const m = res.metrics;

  // Banner meta
  const fileName = uploadedFileName || (selectedPresetId ? selectedPresetId + '_noisy.wav' : 'mic_recording.wav');
  document.getElementById('resultFileName').textContent = fileName;
  document.getElementById('resultDuration').textContent = `${m.duration_seconds}s`;

  // ── Audio Players ──
  const enhancedPlayer = document.getElementById('enhancedPlayer');
  const originalPlayer = document.getElementById('originalPlayer');
  const refPlayer = document.getElementById('referencePlayer');
  const dlBtn = document.getElementById('downloadBtn');
  const refSection = document.getElementById('refPlayerSection');

  if (enhancedPlayer) enhancedPlayer.src = res.enhanced_audio_url || res.enhanced_audio_base64;
  if (originalPlayer) originalPlayer.src = res.input_audio_url || res.input_audio_base64;

  // Reference player: only show if clean reference exists
  if (m.has_clean_reference && (res.reference_audio_url || res.reference_audio_base64)) {
    if (refPlayer) refPlayer.src = res.reference_audio_url || res.reference_audio_base64;
    if (refSection) refSection.style.display = '';
  } else {
    if (refSection) refSection.style.display = 'none';
  }

  if (dlBtn) {
    dlBtn.href = res.enhanced_audio_url || res.enhanced_audio_base64;
    dlBtn.style.display = 'inline-flex';
  }

  // Reset A/B to enhanced
  isShowingEnhanced = true;
  if (enhancedPlayer) enhancedPlayer.style.display = '';
  if (originalPlayer) originalPlayer.style.display = 'none';
  const tabE = document.getElementById('playerTabEnhanced');
  const tabO = document.getElementById('playerTabOriginal');
  if (tabE) tabE.classList.add('active');
  if (tabO) tabO.classList.remove('active');

  // ── Performance Scores ──
  // Check if we have clean references
  const hasRef = m.has_clean_reference;
  
  // SNR
  const snrEl = document.getElementById('scoreSNR');
  const snrIcon = document.getElementById('iconSNR');
  const labelSNR = document.getElementById('labelSNR');
  
  if (labelSNR) {
    labelSNR.textContent = hasRef ? 'SNR' : 'Est. SNR';
    labelSNR.title = hasRef ? 'Signal-to-Noise Ratio' : 'Estimated SNR (no clean reference)';
  }

  if (m.si_snr_improvement_db !== null && m.si_snr_improvement_db !== undefined) {
    const snrVal = Math.abs(m.si_snr_improvement_db);
    const met = snrVal >= 15;
    snrEl.textContent = `+${snrVal.toFixed(1)} dB`;
    snrEl.className = 'score-value ' + (met ? 'met-target' : 'missed-target');
    snrIcon.textContent = met ? '✓' : '✕';
    snrIcon.className = 'score-passfail-icon ' + (met ? 'met-target' : 'missed-target');
  } else {
    snrEl.textContent = 'Ref req';
    snrEl.className = 'score-value ref-required';
    snrIcon.textContent = '';
  }

  // STOI
  const stoiEl = document.getElementById('scoreSTOI');
  const stoiIcon = document.getElementById('iconSTOI');
  if (hasRef && m.stoi_output !== null && m.stoi_output !== undefined) {
    const met = m.stoi_output >= 0.85;
    stoiEl.textContent = m.stoi_output.toFixed(3);
    stoiEl.className = 'score-value ' + (met ? 'met-target' : 'missed-target');
    stoiIcon.textContent = met ? '✓' : '✕';
    stoiIcon.className = 'score-passfail-icon ' + (met ? 'met-target' : 'missed-target');
  } else {
    stoiEl.textContent = 'Ref req';
    stoiEl.className = 'score-value ref-required';
    stoiIcon.textContent = '';
  }

  // PESQ
  const pesqEl = document.getElementById('scorePESQ');
  const pesqIcon = document.getElementById('iconPESQ');
  if (hasRef && m.pesq_output !== null && m.pesq_output !== undefined) {
    const met = m.pesq_output >= 2.5;
    pesqEl.textContent = m.pesq_output.toFixed(2);
    pesqEl.className = 'score-value ' + (met ? 'met-target' : 'missed-target');
    pesqIcon.textContent = met ? '✓' : '✕';
    pesqIcon.className = 'score-passfail-icon ' + (met ? 'met-target' : 'missed-target');
  } else {
    pesqEl.textContent = 'Ref req';
    pesqEl.className = 'score-value ref-required';
    pesqIcon.textContent = '';
  }

  // ── Quick Metrics ──
  // Noise reduction: show as positive magnitude (how many dB of noise removed)
  const nrValue = Math.abs(m.estimated_attenuation_db);
  const nrEl = document.getElementById('metricNoiseReduction');
  nrEl.textContent = nrValue.toFixed(1);
  nrEl.className = 'metric-value ' + (nrValue >= 15 ? 'positive' : '');

  document.getElementById('metricRmsChange').textContent = m.rms_change_db !== undefined ? m.rms_change_db.toFixed(1) : '—';
  document.getElementById('metricPeakChange').textContent = m.peak_change_db !== undefined ? m.peak_change_db.toFixed(1) : '—';
  
  const totalMs = m.total_processing_ms || 1.0;
  document.getElementById('metricLatency').textContent = totalMs < 10 ? totalMs.toFixed(1) : Math.round(totalMs);

  // ── Processing Details ──
  const durationSec = m.duration_seconds || 1.0;
  const rtRatio = totalMs / (durationSec * 1000); // true real-time ratio calculation

  document.getElementById('resultModeBadge').textContent = res.mode || 'hybrid';
  document.getElementById('resultModelName').textContent = res.model_name || '—';
  document.getElementById('resultTotalTime').textContent = `${totalMs < 10 ? totalMs.toFixed(1) : Math.round(totalMs)} ms`;
  document.getElementById('resultRtRatio').textContent = `${rtRatio.toFixed(2)}x ${rtRatio < 1.0 ? '(real-time capable)' : '(offline only)'}`;

  // ── Latency breakdown ──
  document.getElementById('latencyTotal').textContent = `Total: ${totalMs < 10 ? totalMs.toFixed(1) : Math.round(totalMs)} ms`;
  document.getElementById('valCapture').textContent = `${m.capture_ms.toFixed(1)}ms`;
  document.getElementById('valAnc').textContent = `${m.anc_ms.toFixed(1)}ms`;
  document.getElementById('valNeural').textContent = `${m.ai_ms.toFixed(1)}ms`;
  document.getElementById('valPlayback').textContent = `${m.playback_ms.toFixed(1)}ms`;

  document.getElementById('segCapture').style.width = `${(m.capture_ms / totalMs * 100).toFixed(1)}%`;
  document.getElementById('segAnc').style.width = `${(m.anc_ms / totalMs * 100).toFixed(1)}%`;
  document.getElementById('segNeural').style.width = `${(m.ai_ms / totalMs * 100).toFixed(1)}%`;
  document.getElementById('segPlayback').style.width = `${(m.playback_ms / totalMs * 100).toFixed(1)}%`;
  
  setSystemStatus('complete');

  // ── Spectrograms ──
  setSpectrogram('inputSpecImg', 'inputSpecPlaceholder', res.input_spectrogram_base64);
  setSpectrogram('outputSpecImg', 'outputSpecPlaceholder', res.output_spectrogram_base64);

  // ── Charts & Module Analytics ──
  renderVisuals(res.visual_data);
  renderModule3(res.module3);
  renderModule4(res.module4);
  renderModule5(res.module5);
}

function setSpectrogram(imgId, placeholderId, b64) {
  const img = document.getElementById(imgId);
  const ph = document.getElementById(placeholderId);
  if (img && b64) {
    img.src = b64;
    img.style.display = 'block';
    if (ph) ph.style.display = 'none';
  }
}

/* ── CHART RENDERING ── */
const chartColors = {
  text: '#475569',
  grid: 'rgba(0, 0, 0, 0.06)',
  blue: '#2563eb',
  green: '#059669',
  amber: '#d97706',
  red: '#dc2626',
  violet: '#7c3aed',
  cyan: '#0891b2',
};

function createLineChart(canvasId, data, xTitle, yTitle) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'line',
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: xTitle, color: chartColors.text, font: { size: 13, family: 'Inter' } },
          ticks: { color: chartColors.text, font: { size: 12 } },
          grid: { color: chartColors.grid },
        },
        y: {
          title: { display: true, text: yTitle, color: chartColors.text, font: { size: 13, family: 'Inter' } },
          ticks: { color: chartColors.text, font: { size: 12 } },
          grid: { color: chartColors.grid },
        }
      },
      plugins: {
        legend: { labels: { color: chartColors.text, font: { size: 12, family: 'Inter' } } }
      }
    }
  });
}

function createBarChart(canvasId, data, xTitle, yTitle) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: xTitle, color: chartColors.text, font: { size: 13, family: 'Inter' } },
          ticks: { color: chartColors.text, font: { size: 12 } },
          grid: { color: chartColors.grid },
        },
        y: {
          title: { display: true, text: yTitle, color: chartColors.text, font: { size: 13, family: 'Inter' } },
          ticks: { color: chartColors.text, font: { size: 12 } },
          grid: { color: chartColors.grid },
        }
      },
      plugins: {
        legend: { labels: { color: chartColors.text, font: { size: 12, family: 'Inter' } } }
      }
    }
  });
}

/* ── VISUAL CHARTS ── */
function renderVisuals(vis) {
  if (!vis) return;

  createLineChart('chartWaveform', {
    labels: vis.time_s.map(t => `${t}s`),
    datasets: [
      { label: 'Input (Noisy)', data: vis.input_waveform, borderColor: chartColors.amber, borderWidth: 1.5, pointRadius: 0, fill: false },
      { label: 'Enhanced Output', data: vis.output_waveform, borderColor: chartColors.green, borderWidth: 1.5, pointRadius: 0, fill: false }
    ]
  }, 'Time (seconds)', 'Amplitude');

  createLineChart('chartSpectrum', {
    labels: vis.freq_khz.map(f => `${f} kHz`),
    datasets: [
      { label: 'Input Spectrum', data: vis.input_spectrum_db, borderColor: chartColors.red, borderWidth: 1.5, pointRadius: 0, fill: false },
      { label: 'Enhanced Spectrum', data: vis.output_spectrum_db, borderColor: chartColors.blue, borderWidth: 1.5, pointRadius: 0, fill: false }
    ]
  }, 'Frequency (kHz)', 'Power (dB)');
}

/* ── MODULE 3 ── */
function renderModule3(mod3) {
  if (!mod3) return;

  document.getElementById('mod3InitErr').textContent = mod3.initial_wiener_error ? mod3.initial_wiener_error.toFixed(4) : '0.0000';
  document.getElementById('mod3FinalErr').textContent = mod3.final_wiener_error ? mod3.final_wiener_error.toFixed(4) : '0.0000';
  document.getElementById('mod3Ratio').textContent = mod3.wiener_error_ratio ? mod3.wiener_error_ratio.toFixed(4) : '0.0000';

  const decisionEl = document.getElementById('mod3Decision');
  if (decisionEl) {
    decisionEl.innerHTML = mod3.moved_closer_to_wiener
      ? '<span class="badge badge--green">Converged to Wiener</span>'
      : '<span class="badge badge--red">Diverged</span>';
  }

  createLineChart('chartMod3Mse', {
    labels: mod3.lms_mse_curve.map((_, i) => i * 10),
    datasets: [
      { label: 'LMS MSE', data: mod3.lms_mse_curve, borderColor: chartColors.amber, borderWidth: 2, fill: false },
      { label: 'NLMS MSE', data: mod3.nlms_mse_curve, borderColor: chartColors.green, borderWidth: 2, fill: false }
    ]
  }, 'Sample Iterations', 'Mean Squared Error');

  createBarChart('chartMod3Coeffs', {
    labels: mod3.wiener_coeffs.map((_, i) => `Tap ${i}`),
    datasets: [
      { label: 'Wiener Optimal', data: mod3.wiener_coeffs, backgroundColor: 'rgba(37, 99, 235, 0.6)' },
      { label: 'LMS Final', data: mod3.lms_final_coeffs, backgroundColor: 'rgba(217, 119, 6, 0.6)' }
    ]
  }, 'Filter Taps', 'Weight');
}

/* ── MODULE 4 ── */
function renderModule4(mod4) {
  if (!mod4) return;

  document.getElementById('mod4ResDirect').textContent = mod4.residual_power_ratio_direct ? mod4.residual_power_ratio_direct.toFixed(4) : '0.0000';
  document.getElementById('mod4ResFxnlms').textContent = mod4.residual_power_ratio_fxnlms ? mod4.residual_power_ratio_fxnlms.toFixed(4) : '0.0000';
  document.getElementById('mod4ResMismatch').textContent = mod4.residual_power_ratio_mismatched ? mod4.residual_power_ratio_mismatched.toFixed(4) : '0.0000';

  createLineChart('chartMod4Mse', {
    labels: mod4.fxnlms_mse.map((_, i) => i * 10),
    datasets: [
      { label: 'Direct LMS', data: mod4.direct_lms_mse, borderColor: chartColors.red, borderWidth: 2, fill: false },
      { label: 'FxNLMS (Matched)', data: mod4.fxnlms_mse, borderColor: chartColors.green, borderWidth: 2, fill: false },
      { label: 'FxNLMS (Mismatched)', data: mod4.mismatched_fxnlms_mse, borderColor: chartColors.amber, borderWidth: 2, fill: false }
    ]
  }, 'Sample Iterations', 'Error Power');

  createBarChart('chartMod4Paths', {
    labels: mod4.primary_path_impulse.map((_, i) => `Tap ${i}`),
    datasets: [
      { label: 'Primary P(z)', data: mod4.primary_path_impulse, backgroundColor: 'rgba(37, 99, 235, 0.7)' },
      { label: 'Secondary S(z)', data: mod4.secondary_path_impulse, backgroundColor: 'rgba(5, 150, 105, 0.7)' }
    ]
  }, 'Acoustic Taps', 'Amplitude');
}

/* ── MODULE 5 ── */
function renderModule5(mod5) {
  if (!mod5) return;

  document.getElementById('mod5Rmse').textContent = mod5.impulse_rmse ? mod5.impulse_rmse.toFixed(5) : '0.00000';
  document.getElementById('mod5RelErr').textContent = mod5.relative_impulse_error ? mod5.relative_impulse_error.toFixed(5) : '0.00000';
  document.getElementById('mod5MagRmse').textContent = mod5.magnitude_rmse_db ? `${mod5.magnitude_rmse_db.toFixed(2)} dB` : '0.00 dB';
  document.getElementById('mod5PhaseRmse').textContent = mod5.phase_rmse_rad ? `${mod5.phase_rmse_rad.toFixed(2)} rad` : '0.00 rad';

  createLineChart('chartMod5Impulse', {
    labels: mod5.true_impulse.map((_, i) => `Tap ${i}`),
    datasets: [
      { label: 'True S(z)', data: mod5.true_impulse, borderColor: chartColors.blue, borderWidth: 2, fill: false },
      { label: 'Estimated Ŝ(z)', data: mod5.estimated_impulse, borderColor: chartColors.green, borderWidth: 2, borderDash: [4, 4], fill: false }
    ]
  }, 'Filter Taps', 'Impulse Response');

  createLineChart('chartMod5Mag', {
    labels: mod5.freq_axis_khz.map(f => f.toFixed(1)),
    datasets: [
      { label: 'True Magnitude', data: mod5.true_mag_db, borderColor: chartColors.blue, borderWidth: 2, fill: false },
      { label: 'Estimated Magnitude', data: mod5.est_mag_db, borderColor: chartColors.amber, borderWidth: 2, borderDash: [3, 3], fill: false }
    ]
  }, 'Frequency (kHz)', 'Magnitude (dB)');
}
