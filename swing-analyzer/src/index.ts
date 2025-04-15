// Setup TensorFlow.js
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-converter';

// Import all other modules
import './main';

// Initialize TensorFlow backend
async function initialize() {
  console.log("Initializing TensorFlow.js...");
  
  try {
    // Check available backends
    const backends = Object.keys(tf.engine().registryFactory);
    console.log("Available TensorFlow backends:", backends);
    
    // Force WebGL backend for best performance
    await tf.setBackend('webgl');
    const currentBackend = tf.getBackend();
    
    console.log(`TensorFlow.js backend initialized: ${currentBackend}`);
    console.log(`WebGL version: ${tf.env().getNumber('WEBGL_VERSION')}`);
    console.log(`Device pixel ratio: ${window.devicePixelRatio}`);
    
    // Report if we couldn't use WebGL
    if (currentBackend !== 'webgl') {
      console.warn(`WebGL not available, using ${currentBackend} instead. Performance may be affected.`);
    }
  } catch (err) {
    console.error("Failed to initialize TensorFlow backend:", err);
    // Try fallback to CPU as last resort
    try {
      await tf.setBackend('cpu');
      console.warn("Fallback to CPU backend. Performance will be severely limited.");
    } catch (cpuErr) {
      console.error("Failed to initialize any TensorFlow backend:", cpuErr);
    }
  }
}

// Start initialization
initialize().catch(err => {
  console.error("Fatal error during initialization:", err);
}); 