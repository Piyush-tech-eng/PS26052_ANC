/**
 * PS26052 ANC Tactical Command Center & DSP Analytics Dashboard
 */

let currentSourceTab = 'preset';
let selectedPresetId = null;
let uploadedFileBytesB64 = null;
let uploadedFileName = null;
let micMediaRecorder = null;
let micAudioChunks = [];
let micRecordingB64 = null;
let micTimerInterval = null;

let isPlayingClean = false;
let chartInstances = {};

// On Page Load
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  fetchHardwareInfo();
  fetchPresets();
});

// Theme Management
function initTheme() {
  const savedTheme = localStorage.getItem('ps26052_theme');
  if (savedTheme === 'light') {
    document.body.classList.add('light-theme');
    updateThemeUI(true);
  } else {
    document.body.classList.remove('light-theme');
    updateThemeUI(false);
  }
}

function toggleTheme() {
  const isLight = document.body.classList.toggle('light-theme');
  localStorage.setItem('ps26052_theme', isLight ? 'light' : 'dark');
  updateThemeUI(isLight);
  updateChartColors();
}

function updateThemeUI(isLight) {
  const label = document.getElementById('themeToggleLabel');
  const sunIcon = document.querySelector('.sun-icon');
  const moonIcon = document.querySelector('.moon-icon');

  if (label) label.textContent = isLight ? 'ENGINEERING LIGHT' : 'MILITARY HUD';
  if (sunIcon) sunIcon.style.display = isLight ? 'none' : 'inline';
  if (moonIcon) moonIcon.style.display = isLight ? 'inline' : 'none';
}

// Source Tab Switching
function switchSourceTab(tab) {
  currentSourceTab = tab;
  ['preset', 'upload', 'mic'].forEach(t => {
    const btn = document.getElementById(`tab${t.charAt(0).toUpperCase() + t.slice(1)}Btn`);
    const content = document.getElementById(`tab${t.charAt(0).toUpperCase() + t.slice(1)}Content`);
    if (btn) btn.classList.toggle('active', t === tab);
    if (content) content.classList.toggle('active', t === tab);
  });

  const sourceBadge = document.getElementById('selectedSourceBadge');
  if (sourceBadge) {
    sourceBadge.textContent = tab.toUpperCase();
  }
}

// Analytics Navigation Tab Switching
function switchAnalyticsTab(tabName) {
  ['overview', 'mod3', 'mod4', 'mod5', 'visuals'].forEach(t => {
    const btn = document.getElementById(`tabNav${t.charAt(0).toUpperCase() + t.slice(1)}`);
    const pane = document.getElementById(`pane${t.charAt(0).toUpperCase() + t.slice(1)}`);
    if (btn) btn.classList.toggle('active', t === tabName);
    if (pane) pane.style.display = (t === tabName) ? 'flex' : 'none';
  });

  // Resize active Chart.js charts
  Object.values(chartInstances).forEach(chart => {
    if (chart) chart.resize();
  });
}

// Hardware Telemetry API
async function fetchHardwareInfo() {
  try {
    const res = await fetch('/api/hardware');
    const data = await res.json();
    const platBadge = document.getElementById('platformBadge');
    if (platBadge && data.os) {
      platBadge.textContent = `${data.os} (Python ${data.python})`;
    }
  } catch (err) {
    console.error('Failed to fetch hardware info:', err);
  }
}

// Presets API
async function fetchPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    const select = document.getElementById('presetSelect');
    if (!select || !data.presets) return;

    select.innerHTML = '<option value="" disabled selected>Select a defence scenario...</option>';
    data.presets.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.name;
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

  const infoBox = document.getElementById('presetDetailsText');
  if (infoBox && selectedPresetId) {
    infoBox.textContent = `Preset '${selectedPresetId}' selected. Dual-channel reference audio prepared.`;
  }
}

// File Upload Handling
function handleFileSelected(event) {
  const file = event.target.files[0];
  if (!file) return;

  uploadedFileName = file.name;
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadedFileBytesB64 = e.target.result;
    const banner = document.getElementById('fileLoadedBanner');
    const nameSpan = document.getElementById('loadedFileName');
    const sizeSpan = document.getElementById('loadedFileSize');

    if (nameSpan) nameSpan.textContent = file.name;
    if (sizeSpan) sizeSpan.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    if (banner) banner.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

// Drag & Drop Setup
const dropzone = document.getElementById('fileDropzone');
if (dropzone) {
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, e => e.preventDefault(), false);
  });

  dropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      document.getElementById('audioFileInput').files = files;
      handleFileSelected({ target: { files } });
    }
  });
}

// Live Mic Recording
async function toggleMicRecording() {
  const btn = document.getElementById('micRecordBtn');
  const btnText = document.getElementById('micRecordBtnText');
  const timer = document.getElementById('recordTimer');

  if (micMediaRecorder && micMediaRecorder.state === 'recording') {
    micMediaRecorder.stop();
    btnText.textContent = 'Start Recording';
    btn.classList.remove('recording');
    clearInterval(micTimerInterval);
  } else {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micMediaRecorder = new MediaRecorder(stream);
      micAudioChunks = [];

      micMediaRecorder.ondataavailable = e => micAudioChunks.push(e.data);
      micMediaRecorder.onstop = async () => {
        const audioBlob = new Blob(micAudioChunks, { type: 'audio/wav' });
        const reader = new FileReader();
        reader.onload = e => {
          micRecordingB64 = e.target.result;
          if (timer) timer.textContent = 'REC DONE';
        };
        reader.readAsDataURL(audioBlob);
      };

      micMediaRecorder.start();
      btnText.textContent = 'Stop Recording';
      btn.classList.add('recording');

      let seconds = 0;
      micTimerInterval = setInterval(() => {
        seconds += 0.1;
        if (timer) timer.textContent = `00:0${seconds.toFixed(1)}s`;
      }, 100);

    } catch (err) {
      alert('Microphone access denied or unavailable: ' + err.message);
    }
  }
}

function onModeChanged() {
  const mode = document.getElementById('pipelineModeSelect').value;
  const filterGroup = document.getElementById('filterTapsGroup');
  const stepGroup = document.getElementById('stepSizeGroup');
  const modelGroup = document.getElementById('modelSelectGroup');

  if (filterGroup) filterGroup.style.display = (mode === 'ai_only') ? 'none' : 'flex';
  if (stepGroup) stepGroup.style.display = (mode === 'ai_only') ? 'none' : 'flex';
  if (modelGroup) modelGroup.style.display = (mode === 'anc_only') ? 'none' : 'flex';
}

// Spectrogram Sub-Tab
function switchSpectrogramTab(type) {
  const sideBtn = document.getElementById('specTabSideBtn');
  const diffBtn = document.getElementById('specTabDiffBtn');
  const dualView = document.getElementById('spectrogramDualView');
  const diffView = document.getElementById('spectrogramDiffView');

  if (sideBtn) sideBtn.classList.toggle('active', type === 'side');
  if (diffBtn) diffBtn.classList.toggle('active', type === 'diff');
  if (dualView) dualView.style.display = (type === 'side') ? 'grid' : 'none';
  if (diffView) diffView.style.display = (type === 'diff') ? 'block' : 'none';
}

// Main Execution Pipeline Call
async function runProcessingPipeline() {
  const btn = document.getElementById('executeBtn');
  const spinner = document.getElementById('executeSpinner');
  const btnText = document.getElementById('executeBtnText');

  if (btn) btn.disabled = true;
  if (spinner) spinner.style.display = 'inline-block';
  if (btnText) btnText.textContent = 'PROCESSING DSP & AI ENGINE...';

  const mode = document.getElementById('pipelineModeSelect').value;
  const modelName = document.getElementById('modelSelect').value;
  const filterLength = parseInt(document.getElementById('filterTapsSelect').value, 10);
  const stepSize = parseFloat(document.getElementById('stepSizeSelect').value);

  const payload = {
    mode,
    model_name: modelName,
    filter_length: filterLength,
    step_size: stepSize,
  };

  if (currentSourceTab === 'preset') {
    if (!selectedPresetId) {
      alert('Please select a preset audio scenario.');
      resetExecuteBtn();
      return;
    }
    payload.preset_id = selectedPresetId;
  } else if (currentSourceTab === 'upload') {
    if (!uploadedFileBytesB64) {
      alert('Please upload an audio file first.');
      resetExecuteBtn();
      return;
    }
    payload.audio_base64 = uploadedFileBytesB64;
  } else if (currentSourceTab === 'mic') {
    if (!micRecordingB64) {
      alert('Please record audio using the microphone first.');
      resetExecuteBtn();
      return;
    }
    payload.audio_base64 = micRecordingB64;
  }

  try {
    const res = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const result = await res.json();
    if (!result.success) {
      alert('Processing Error: ' + (result.error || result.message));
      resetExecuteBtn();
      return;
    }

    renderResults(result);

  } catch (err) {
    alert('Server communication failed: ' + err.message);
  } finally {
    resetExecuteBtn();
  }
}

function resetExecuteBtn() {
  const btn = document.getElementById('executeBtn');
  const spinner = document.getElementById('executeSpinner');
  const btnText = document.getElementById('executeBtnText');

  if (btn) btn.disabled = false;
  if (spinner) spinner.style.display = 'none';
  if (btnText) btnText.textContent = 'EXECUTE NOISE CANCELLATION';
}

// Render Results Data
function renderResults(res) {
  const m = res.metrics;

  // Overview HUD Metrics
  document.getElementById('metricAttenuation').textContent = `-${m.estimated_attenuation_db.toFixed(1)} dB`;
  document.getElementById('metricSiSnr').textContent = (m.si_snr_improvement_db !== null) ? `+${m.si_snr_improvement_db.toFixed(1)} dB` : 'N/A';
  document.getElementById('metricStoi').textContent = (m.stoi_input !== null && m.stoi_output !== null) ? `${m.stoi_input.toFixed(2)} → ${m.stoi_output.toFixed(2)}` : 'N/A';
  document.getElementById('metricRtRatio').textContent = `${m.realtime_ratio.toFixed(2)}x`;

  const rtBadge = document.getElementById('rtRatioBadge');
  if (rtBadge) {
    rtBadge.textContent = `${m.realtime_ratio < 1.0 ? '< 1.0x REAL-TIME READY' : 'OFFLINE LATENCY'}`;
    rtBadge.className = `badge ${m.realtime_ratio < 1.0 ? 'badge-emerald' : 'badge-cyan'}`;
  }

  // Latency Bar
  const totalMs = m.total_processing_ms || 1.0;
  document.getElementById('latencyTotalVal').textContent = `Total: ${totalMs.toFixed(1)} ms`;
  document.getElementById('valLatCap').textContent = `${m.capture_ms.toFixed(1)}ms`;
  document.getElementById('valLatAnc').textContent = `${m.anc_ms.toFixed(1)}ms`;
  document.getElementById('valLatAi').textContent = `${m.ai_ms.toFixed(1)}ms`;
  document.getElementById('valLatPlay').textContent = `${m.playback_ms.toFixed(1)}ms`;

  document.getElementById('segCap').style.width = `${(m.capture_ms / totalMs * 100).toFixed(1)}%`;
  document.getElementById('segAnc').style.width = `${(m.anc_ms / totalMs * 100).toFixed(1)}%`;
  document.getElementById('segAi').style.width = `${(m.ai_ms / totalMs * 100).toFixed(1)}%`;
  document.getElementById('segPlay').style.width = `${(m.playback_ms / totalMs * 100).toFixed(1)}%`;

  // Audio Players
  const inPlayer = document.getElementById('inputAudioPlayer');
  const outPlayer = document.getElementById('outputAudioPlayer');
  const refPlayer = document.getElementById('refAudioPlayer');
  const dlBtn = document.getElementById('downloadEnhancedBtn');
  const abBtn = document.getElementById('abToggleBtn');

  if (inPlayer) inPlayer.src = res.input_audio_url || res.input_audio_base64;
  if (outPlayer) outPlayer.src = res.enhanced_audio_url || res.enhanced_audio_base64;
  if (refPlayer) refPlayer.src = res.reference_audio_url || res.reference_audio_base64 || '';

  if (dlBtn) {
    dlBtn.href = res.enhanced_audio_url || res.enhanced_audio_base64;
    dlBtn.style.display = 'inline-flex';
  }
  if (abBtn) abBtn.disabled = false;

  // Spectrograms
  const inImg = document.getElementById('inputSpectrogramImg');
  const outImg = document.getElementById('outputSpectrogramImg');
  const diffImg = document.getElementById('diffSpectrogramImg');

  if (inImg) { inImg.src = res.input_spectrogram_base64; inImg.style.display = 'block'; }
  if (outImg) { outImg.src = res.output_spectrogram_base64; outImg.style.display = 'block'; }
  if (diffImg) { diffImg.src = res.difference_spectrogram_base64; diffImg.style.display = 'block'; }

  document.getElementById('inputSpecPlaceholder').style.display = 'none';
  document.getElementById('outputSpecPlaceholder').style.display = 'none';
  document.getElementById('diffSpecPlaceholder').style.display = 'none';

  // Render Charts for Module 3, Module 4, Module 5, & Visuals
  renderModule3(res.module3);
  renderModule4(res.module4);
  renderModule5(res.module5);
  renderVisuals(res.visual_data);
}

// Instant A/B Toggle
function toggleABPlayback() {
  const inPlayer = document.getElementById('inputAudioPlayer');
  const outPlayer = document.getElementById('outputAudioPlayer');
  const btnText = document.getElementById('abToggleText');

  if (isPlayingClean) {
    outPlayer.pause();
    inPlayer.currentTime = outPlayer.currentTime;
    inPlayer.play();
    btnText.textContent = 'Instant A/B: Switch to Enhanced Speech';
    isPlayingClean = false;
  } else {
    inPlayer.pause();
    outPlayer.currentTime = inPlayer.currentTime;
    outPlayer.play();
    btnText.textContent = 'Instant A/B: Switch to Noisy Input';
    isPlayingClean = true;
  }
}

// Module 3 Chart Renderer
function renderModule3(mod3) {
  if (!mod3) return;

  document.getElementById('mod3InitErr').textContent = mod3.initial_wiener_error ? mod3.initial_wiener_error.toFixed(4) : '0.0000';
  document.getElementById('mod3FinalErr').textContent = mod3.final_wiener_error ? mod3.final_wiener_error.toFixed(4) : '0.0000';
  document.getElementById('mod3Ratio').textContent = mod3.wiener_error_ratio ? mod3.wiener_error_ratio.toFixed(4) : '0.0000';
  
  const badge = document.getElementById('mod3DecisionBadge');
  if (badge) {
    badge.textContent = mod3.moved_closer_to_wiener ? 'CONVERGED TO WIENER' : 'DIVERGED';
    badge.className = `badge ${mod3.moved_closer_to_wiener ? 'badge-emerald' : 'badge-red'}`;
  }

  // MSE Curve Chart
  createLineChart('chartMod3Mse', {
    labels: mod3.lms_mse_curve.map((_, i) => i * 10),
    datasets: [
      { label: 'Standard LMS MSE', data: mod3.lms_mse_curve, borderColor: '#f59e0b', borderWidth: 2, fill: false },
      { label: 'Normalized NLMS MSE', data: mod3.nlms_mse_curve, borderColor: '#10b981', borderWidth: 2, fill: false }
    ]
  }, 'Sample Iterations', 'Mean Squared Error');

  // Coeffs Comparison Chart
  createBarChart('chartMod3Coeffs', {
    labels: mod3.wiener_coeffs.map((_, i) => `Tap ${i}`),
    datasets: [
      { label: 'Wiener Optimal Weights', data: mod3.wiener_coeffs, backgroundColor: 'rgba(56, 189, 248, 0.6)' },
      { label: 'LMS Final Taps', data: mod3.lms_final_coeffs, backgroundColor: 'rgba(245, 158, 11, 0.6)' }
    ]
  }, 'Filter Taps', 'Weight Magnitude');
}

// Module 4 Chart Renderer
function renderModule4(mod4) {
  if (!mod4) return;

  document.getElementById('mod4ResDirect').textContent = mod4.residual_power_ratio_direct ? mod4.residual_power_ratio_direct.toFixed(4) : '0.0000';
  document.getElementById('mod4ResFxnlms').textContent = mod4.residual_power_ratio_fxnlms ? mod4.residual_power_ratio_fxnlms.toFixed(4) : '0.0000';
  document.getElementById('mod4ResMismatched').textContent = mod4.residual_power_ratio_mismatched ? mod4.residual_power_ratio_mismatched.toFixed(4) : '0.0000';

  // FxNLMS Error Power Chart
  createLineChart('chartMod4Mse', {
    labels: mod4.fxnlms_mse.map((_, i) => i * 10),
    datasets: [
      { label: 'Direct LMS (No S(z) Model)', data: mod4.direct_lms_mse, borderColor: '#ef4444', borderWidth: 2, fill: false },
      { label: 'FxNLMS (Matched S(z) Model)', data: mod4.fxnlms_mse, borderColor: '#10b981', borderWidth: 2, fill: false },
      { label: 'FxNLMS (Mismatched S(z) Model)', data: mod4.mismatched_fxnlms_mse, borderColor: '#f59e0b', borderWidth: 2, fill: false }
    ]
  }, 'Sample Iterations', 'Error Power e^2[n]');

  // Impulse Paths Chart
  createBarChart('chartMod4Paths', {
    labels: mod4.primary_path_impulse.map((_, i) => `Tap ${i}`),
    datasets: [
      { label: 'Primary Path P(z)', data: mod4.primary_path_impulse, backgroundColor: 'rgba(59, 130, 246, 0.7)' },
      { label: 'Secondary Path S(z)', data: mod4.secondary_path_impulse, backgroundColor: 'rgba(16, 185, 129, 0.7)' }
    ]
  }, 'Acoustic Taps', 'Amplitude');
}

// Module 5 Chart Renderer
function renderModule5(mod5) {
  if (!mod5) return;

  document.getElementById('mod5Rmse').textContent = mod5.impulse_rmse ? mod5.impulse_rmse.toFixed(5) : '0.00000';
  document.getElementById('mod5RelErr').textContent = mod5.relative_impulse_error ? mod5.relative_impulse_error.toFixed(5) : '0.00000';
  document.getElementById('mod5MagRmse').textContent = mod5.magnitude_rmse_db ? `${mod5.magnitude_rmse_db.toFixed(2)} dB` : '0.00 dB';
  document.getElementById('mod5PhaseRmse').textContent = mod5.phase_rmse_rad ? `${mod5.phase_rmse_rad.toFixed(2)} rad` : '0.00 rad';

  // System ID Impulse Comparison
  createLineChart('chartMod5Impulse', {
    labels: mod5.true_impulse.map((_, i) => `Tap ${i}`),
    datasets: [
      { label: 'True Ground Truth S(z)', data: mod5.true_impulse, borderColor: '#38bdf8', borderWidth: 2, fill: false },
      { label: 'Estimated Path S^(z)', data: mod5.estimated_impulse, borderColor: '#10b981', borderWidth: 2, borderDash: [4, 4], fill: false }
    ]
  }, 'Filter Taps', 'Impulse Response');

  // Frequency Magnitude Response Chart
  createLineChart('chartMod5Mag', {
    labels: mod5.freq_axis_khz.map(f => f.toFixed(1)),
    datasets: [
      { label: 'True Path Magnitude (dB)', data: mod5.true_mag_db, borderColor: '#3b82f6', borderWidth: 2, fill: false },
      { label: 'Estimated Path Magnitude (dB)', data: mod5.est_mag_db, borderColor: '#f59e0b', borderWidth: 2, borderDash: [3, 3], fill: false }
    ]
  }, 'Frequency (kHz)', 'Magnitude (dB)');
}

// Visuals Chart Renderer
function renderVisuals(vis) {
  if (!vis) return;

  // Time Domain Waveforms
  createLineChart('chartVisWaveform', {
    labels: vis.time_s.map(t => `${t}s`),
    datasets: [
      { label: 'Input Audio (Disturbed)', data: vis.input_waveform, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0 },
      { label: 'Enhanced Speech Output', data: vis.output_waveform, borderColor: '#10b981', borderWidth: 1.5, pointRadius: 0 }
    ]
  }, 'Time (seconds)', 'Amplitude');

  // Frequency Spectrum
  createLineChart('chartVisSpectrum', {
    labels: vis.freq_khz.map(f => `${f}kHz`),
    datasets: [
      { label: 'Input Spectrum (dB)', data: vis.input_spectrum_db, borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0 },
      { label: 'Enhanced Spectrum (dB)', data: vis.output_spectrum_db, borderColor: '#38bdf8', borderWidth: 1.5, pointRadius: 0 }
    ]
  }, 'Frequency (kHz)', 'Power (dB)');
}

// Chart.js Helper Functions
function createLineChart(canvasId, data, xTitle, yTitle) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
  }

  const isLight = document.body.classList.contains('light-theme');
  const textColor = isLight ? '#475569' : '#94a3b8';
  const gridColor = isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.08)';

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'line',
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: xTitle, color: textColor, font: { size: 10 } },
          ticks: { color: textColor, font: { size: 9 } },
          grid: { color: gridColor },
        },
        y: {
          title: { display: true, text: yTitle, color: textColor, font: { size: 10 } },
          ticks: { color: textColor, font: { size: 9 } },
          grid: { color: gridColor },
        }
      },
      plugins: {
        legend: { labels: { color: textColor, font: { size: 10 } } }
      }
    }
  });
}

function createBarChart(canvasId, data, xTitle, yTitle) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
  }

  const isLight = document.body.classList.contains('light-theme');
  const textColor = isLight ? '#475569' : '#94a3b8';
  const gridColor = isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.08)';

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: xTitle, color: textColor, font: { size: 10 } },
          ticks: { color: textColor, font: { size: 9 } },
          grid: { color: gridColor },
        },
        y: {
          title: { display: true, text: yTitle, color: textColor, font: { size: 10 } },
          ticks: { color: textColor, font: { size: 9 } },
          grid: { color: gridColor },
        }
      },
      plugins: {
        legend: { labels: { color: textColor, font: { size: 10 } } }
      }
    }
  });
}

function updateChartColors() {
  Object.values(chartInstances).forEach(chart => {
    if (chart) chart.update();
  });
}
