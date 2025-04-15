// Import application components
import { SwingAnalyzer } from './SwingAnalyzer';
import { AppState } from './types';

// Get DOM elements
const video = document.getElementById('video') as HTMLVideoElement;
const canvas = document.getElementById('output-canvas') as HTMLCanvasElement;
const cameraBtn = document.getElementById('camera-btn') as HTMLButtonElement;
const playPauseBtn = document.getElementById('play-pause-btn') as HTMLButtonElement;
const stopBtn = document.getElementById('stop-btn') as HTMLButtonElement;
const videoUpload = document.getElementById('video-upload') as HTMLInputElement;
const repCounter = document.getElementById('rep-counter') as HTMLSpanElement;
const status = document.getElementById('status') as HTMLDivElement;
const spineAngle = document.getElementById('spine-angle') as HTMLSpanElement;
const displayModeRadios = document.querySelectorAll('input[name="display-mode"]') as NodeListOf<HTMLInputElement>;

// Application state
const appState: AppState = {
  isModelLoaded: false,
  isProcessing: false,
  usingCamera: false,
  repCounter: {
    count: 0,
    isHinge: false,
    lastHingeState: false,
    hingeThreshold: 45
  },
  showBodyParts: true,
  bodyPartDisplayTime: 0.5 // Show body part labels for 0.5 seconds
};

// Create swing analyzer
let swingAnalyzer: SwingAnalyzer | null = null;

// Initialize the application
async function initApp() {
  updateStatus('Loading model...');
  
  // Setup video and canvas dimensions initially
  setupVideoCanvas();
  
  // Initialize listeners
  setupEventListeners();
  
  updateStatus('Ready. Upload a video or start camera.');
}

function setupVideoCanvas() {
  // Make sure canvas has the same dimensions as video
  function updateDimensions() {
    if (video.videoWidth && video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Make sure video is visible
      video.style.display = 'block';
      canvas.style.display = 'block';
      
      console.log(`Video dimensions set: ${video.videoWidth}x${video.videoHeight}`);
    }
  }
  
  // Add listeners for dimension updates
  video.addEventListener('loadedmetadata', updateDimensions);
  video.addEventListener('resize', updateDimensions);
  
  // Set initial dimensions
  if (video.videoWidth && video.videoHeight) {
    updateDimensions();
  } else {
    // Set default dimensions
    canvas.width = 640;
    canvas.height = 480;
  }
}

function setupEventListeners() {
  // Camera button
  cameraBtn.addEventListener('click', startCamera);
  
  // Play/Pause toggle button
  playPauseBtn.addEventListener('click', togglePlayPause);
  
  // Stop button
  stopBtn.addEventListener('click', stopVideo);
  
  // Video upload
  videoUpload.addEventListener('change', handleVideoUpload);
  
  // Video events
  video.addEventListener('play', () => {
    if (swingAnalyzer) {
      swingAnalyzer.startProcessing();
      appState.isProcessing = true;
      // Update button text
      playPauseBtn.textContent = 'Pause';
    }
  });
  
  video.addEventListener('pause', () => {
    if (swingAnalyzer) {
      swingAnalyzer.stopProcessing();
      appState.isProcessing = false;
      // Update button text
      playPauseBtn.textContent = 'Play';
    }
  });
  
  video.addEventListener('ended', () => {
    if (swingAnalyzer) {
      swingAnalyzer.stopProcessing();
      appState.isProcessing = false;
      // Reset button
      playPauseBtn.textContent = 'Play';
    }
  });
  
  // Add debug controls
  setupDebugControls();
}

function setupDebugControls() {
  displayModeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
      const target = e.target as HTMLInputElement;
      const mode = target.value;
      
      console.log(`Display mode changed to: ${mode}`);
      
      switch (mode) {
        case 'both':
          video.style.opacity = '1';
          canvas.style.display = 'block';
          break;
        case 'video':
          video.style.opacity = '1';
          canvas.style.display = 'none';
          break;
        case 'overlay':
          // Make video transparent but still there for pose detection
          video.style.opacity = '0.1';
          canvas.style.display = 'block';
          // Set canvas background to black
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
          }
          break;
      }
    });
  });
}

async function initializeAnalyzer() {
  if (!swingAnalyzer) {
    swingAnalyzer = new SwingAnalyzer(
      video, 
      canvas, 
      appState.showBodyParts,
      appState.bodyPartDisplayTime
    );
    
    try {
      await swingAnalyzer.initialize();
      appState.isModelLoaded = true;
      updateStatus('Model loaded. Ready to analyze.');
    } catch (error) {
      console.error('Error initializing analyzer:', error);
      updateStatus('Error loading model. Please refresh and try again.');
    }
  }
}

function updateStatus(message: string) {
  if (status) {
    status.textContent = message;
  }
}

function updateButtonStates(
  canCamera: boolean, 
  canPlayPause: boolean, 
  canStop: boolean
) {
  cameraBtn.disabled = !canCamera;
  playPauseBtn.disabled = !canPlayPause;
  stopBtn.disabled = !canStop;
  
  // Update play/pause button text to match video state
  if (canPlayPause) {
    playPauseBtn.textContent = video.paused ? 'Play' : 'Pause';
  }
}

async function startCamera() {
  try {
    // Request camera permissions
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment', // Prefer rear camera on mobile
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });
    
    // Make sure video is visible
    video.style.display = 'block';
    
    // Set video source to camera stream
    video.srcObject = stream;
    appState.usingCamera = true;
    
    // Wait for video metadata to load
    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => {
        console.log(`Camera stream loaded: ${video.videoWidth}x${video.videoHeight}`);
        resolve();
      };
    });
    
    // Initialize analyzer if needed
    await initializeAnalyzer();
    
    // Start video
    await video.play();
    
    // Update UI
    updateButtonStates(false, true, true);
    updateStatus('Camera active. Analyzing motion...');
  } catch (error) {
    console.error('Error accessing camera:', error);
    updateStatus('Camera access denied or not available.');
  }
}

async function handleVideoUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  if (input.files && input.files.length > 0) {
    const file = input.files[0];
    
    // Stop camera if it's active
    if (appState.usingCamera) {
      stopCamera();
    }
    
    // Make sure video is visible
    video.style.display = 'block';
    
    // Create object URL for the video file
    const videoURL = URL.createObjectURL(file);
    video.src = videoURL;
    appState.usingCamera = false;
    
    // Force video to reload and prep for playing
    video.load();
    
    // Wait for video metadata to load
    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => {
        console.log(`Video loaded: ${video.videoWidth}x${video.videoHeight}`);
        resolve();
      };
    });
    
    // Initialize analyzer if needed
    await initializeAnalyzer();
    
    // Update UI
    updateButtonStates(true, true, false);
    updateStatus(`Loaded video: ${file.name}. Press Play to analyze.`);
    
    // Reset rep counter
    if (swingAnalyzer) {
      swingAnalyzer.reset();
    }
  }
}

// Replace the separate play and pause functions with a toggle function
function togglePlayPause() {
  if (video.paused) {
    // Currently paused, so play
    video.play();
    playPauseBtn.textContent = 'Pause';
    updateButtonStates(true, true, true);
  } else {
    // Currently playing, so pause
    video.pause();
    playPauseBtn.textContent = 'Play';
    updateButtonStates(true, true, true);
  }
}

function stopVideo() {
  video.pause();
  
  // Reset video position
  video.currentTime = 0;
  
  if (appState.usingCamera) {
    stopCamera();
  }
  
  updateButtonStates(true, true, false);
  playPauseBtn.textContent = 'Play';
  
  // Reset rep counter
  if (swingAnalyzer) {
    swingAnalyzer.reset();
  }
}

function stopCamera() {
  // Stop camera stream
  if (video.srcObject) {
    const stream = video.srcObject as MediaStream;
    const tracks = stream.getTracks();
    
    tracks.forEach(track => track.stop());
    video.srcObject = null;
  }
  
  appState.usingCamera = false;
  updateButtonStates(true, false, false);
}

// Start the application
document.addEventListener('DOMContentLoaded', initApp);
