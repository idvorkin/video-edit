import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs-core';
import { MediaPipeBodyParts, PoseKeypoint, PoseResult, RepCounter } from './types';

export class SwingAnalyzer {
  private detector: poseDetection.PoseDetector | null = null;
  private video: HTMLVideoElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private rafId: number | null = null;
  private startTime = 0;
  private fps = 0;
  private frameCount = 0;
  private repCounter: RepCounter;
  private showBodyParts: boolean;
  private bodyPartDisplaySeconds: number;
  private frameTimestamp = 0;
  private spineAngle = 0;
  private keypointUpdateCallback: ((keypoints: any[]) => void) | null = null;
  
  constructor(
    video: HTMLVideoElement, 
    canvas: HTMLCanvasElement,
    showBodyParts = true,
    bodyPartDisplaySeconds = 0.5,
    keypointUpdateCallback: ((keypoints: any[]) => void) | null = null
  ) {
    this.video = video;
    this.canvas = canvas;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    this.ctx = ctx;
    this.repCounter = {
      count: 0,
      isHinge: false,
      lastHingeState: false,
      hingeThreshold: 45 // Degrees, matching your Python implementation
    };
    this.showBodyParts = showBodyParts;
    this.bodyPartDisplaySeconds = bodyPartDisplaySeconds;
    this.keypointUpdateCallback = keypointUpdateCallback;
  }

  async initialize(): Promise<void> {
    console.log("Initializing Swing Analyzer...");
    
    // Make sure TensorFlow.js is ready and using WebGL backend
    try {
      await tf.setBackend('webgl');
      console.log(`Using backend: ${tf.getBackend()}`);
      
      // Use MoveNet - better performance on mobile
      const detectorConfig = {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableSmoothing: true,
      };
      
      console.log("Loading pose detection model...");
      try {
        this.detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet, 
          detectorConfig
        );
        
        this.canvas.width = this.video.videoWidth || 640;
        this.canvas.height = this.video.videoHeight || 480;
        
        console.log("Pose detector initialized successfully");
        
        // Verify detector is working by running a basic test
        try {
          // Create a dummy canvas to test with
          const testCanvas = document.createElement('canvas');
          testCanvas.width = 300;
          testCanvas.height = 300;
          const ctx = testCanvas.getContext('2d');
          if (ctx) {
            // Draw a simple shape that might be detected as a person
            ctx.fillStyle = 'white';
            ctx.fillRect(100, 50, 100, 200);
            
            // Try to run detector on test canvas
            const testPoses = await this.detector.estimatePoses(testCanvas);
            console.log(`Test detection complete. Found ${testPoses.length} poses.`);
          }
        } catch (testErr) {
          console.warn("Test detection failed, but continuing:", testErr);
          // Continue anyway as the real video might work
        }
      } catch (modelError) {
        console.error("Failed to initialize primary model:", modelError);
        
        // Try a fallback model
        console.log("Trying fallback model...");
        try {
          this.detector = await poseDetection.createDetector(
            poseDetection.SupportedModels.PoseNet
          );
          console.log("Fallback model initialized");
        } catch (fallbackError) {
          console.error("Failed to initialize fallback model:", fallbackError);
          throw new Error("Could not initialize any pose detection model");
        }
      }
    } catch (error) {
      console.error("Failed to initialize pose detector:", error);
      throw error;
    }
  }

  async detectPose(): Promise<PoseResult | null> {
    if (!this.detector) {
      console.error("Detector not initialized");
      return null;
    }
    
    try {
      console.log("Detecting pose...");
      const poses = await this.detector.estimatePoses(this.video);
      
      if (poses.length === 0) {
        console.log("No poses detected in frame");
        return null;
      }
      
      // Log comprehensive information about the detected pose
      console.log(`Pose detected with ${poses[0].keypoints.length} keypoints`);
      
      // Create a formatted summary of key body parts
      const bodyPartSummary = {
        face: poses[0].keypoints.some(kp => kp.name === 'nose' && (kp.score || 0) > 0.5) ? 'Detected' : 'Not detected',
        shoulders: poses[0].keypoints.filter(kp => 
          (kp.name === 'left_shoulder' || kp.name === 'right_shoulder') && (kp.score || 0) > 0.5
        ).length + '/2 detected',
        hips: poses[0].keypoints.filter(kp => 
          (kp.name === 'left_hip' || kp.name === 'right_hip') && (kp.score || 0) > 0.5
        ).length + '/2 detected',
        score: poses[0].score?.toFixed(2) || 'N/A'
      };
      
      console.log('Body part summary:', bodyPartSummary);
      
      // Log all keypoints for debugging
      console.log('All detected keypoints:', 
        poses[0].keypoints.map(kp => ({
          name: kp.name || 'unknown',
          x: Math.round(kp.x),
          y: Math.round(kp.y),
          confidence: kp.score?.toFixed(2) || 'N/A'
        }))
      );
      
      return {
        keypoints: poses[0].keypoints
      };
    } catch (error) {
      console.error('Error detecting pose:', error);
      return null;
    }
  }
  
  // Calculate spine angle from vertical (0-180 degrees)
  calculateSpineVertical(keypoints: PoseKeypoint[]): number {
    // Log visibility of ALL keypoints, not just a sample
    const allVisibilityReport = keypoints.map((kp, i) => {
      const name = kp.name || this.getBodyPartName(i) || `point_${i}`;
      const confidence = kp.score !== undefined ? kp.score : 
                         kp.visibility !== undefined ? kp.visibility : 0;
      // Format as "Name: value" for better readability in console  
      return `${name}: ${confidence.toFixed(2)}`;
    });
    
    // Log each keypoint on a new line for better readability
    console.log("ALL KEYPOINTS VISIBILITY:");
    console.table(allVisibilityReport);
    
    // Also log the raw keypoints for detailed inspection
    console.log("KEYPOINTS RAW DATA:", keypoints);
    
    // Try multiple approaches to calculate angle in order of preference
    
    // 1. First approach: Use shoulders and hips if available (best)
    const leftShoulder = keypoints[MediaPipeBodyParts.LEFT_SHOULDER];
    const rightShoulder = keypoints[MediaPipeBodyParts.RIGHT_SHOULDER];
    const leftHip = keypoints[MediaPipeBodyParts.LEFT_HIP];
    const rightHip = keypoints[MediaPipeBodyParts.RIGHT_HIP];
    
    // Safe array of points that exist and are visible
    const safeShoulders = [];
    const safeHips = [];
    
    if (leftShoulder && this.isPointVisible(leftShoulder)) safeShoulders.push(leftShoulder);
    if (rightShoulder && this.isPointVisible(rightShoulder)) safeShoulders.push(rightShoulder);
    if (leftHip && this.isPointVisible(leftHip)) safeHips.push(leftHip);
    if (rightHip && this.isPointVisible(rightHip)) safeHips.push(rightHip);
    
    if (safeShoulders.length > 0 && safeHips.length > 0) {
      // Calculate average positions
      const topX = safeShoulders.reduce((sum, p) => sum + p.x, 0) / safeShoulders.length;
      const topY = safeShoulders.reduce((sum, p) => sum + p.y, 0) / safeShoulders.length;
      const bottomX = safeHips.reduce((sum, p) => sum + p.x, 0) / safeHips.length;
      const bottomY = safeHips.reduce((sum, p) => sum + p.y, 0) / safeHips.length;
      
      // Calculate angle from vertical axis
      const deltaX = topX - bottomX;
      const deltaY = bottomY - topY; // Inverted because Y axis points down in screen coordinates
      
      const angle = Math.abs(Math.atan2(deltaX, deltaY) * 180 / Math.PI);
      console.log(`Spine angle calculated (shoulders-hips): ${angle.toFixed(2)}°`);
      return angle;
    }
    
    // 2. Second approach: Use face orientation as fallback
    const nose = keypoints[MediaPipeBodyParts.NOSE];
    const leftEye = keypoints[MediaPipeBodyParts.LEFT_EYE];
    const rightEye = keypoints[MediaPipeBodyParts.RIGHT_EYE];
    
    if (nose && (leftEye || rightEye) && this.isPointVisible(nose)) {
      // Use visible eye, or average of both if available
      let eyeX, eyeY;
      
      if (leftEye && rightEye && this.isPointVisible(leftEye) && this.isPointVisible(rightEye)) {
        eyeX = (leftEye.x + rightEye.x) / 2;
        eyeY = (leftEye.y + rightEye.y) / 2;
      } else if (leftEye && this.isPointVisible(leftEye)) {
        eyeX = leftEye.x;
        eyeY = leftEye.y;
      } else if (rightEye && this.isPointVisible(rightEye)) {
        eyeX = rightEye.x;
        eyeY = rightEye.y;
      } else {
        // No eyes visible
        console.log("No visible eyes found for angle calculation");
        return 0;
      }
      
      // Calculate angle of face from vertical
      // This assumes head tilt correlates with body tilt
      const deltaX = eyeX - nose.x;
      const deltaY = nose.y - eyeY; // Inverted because Y axis points down
      
      // Add 90 degrees to convert from face orientation to body orientation (approximate)
      const faceAngle = Math.atan2(deltaX, deltaY) * 180 / Math.PI;
      // Map face angle to spine angle - this is a rough approximation
      // We need to adjust because face and spine angles have different reference points
      let spineAngle = Math.abs(faceAngle) * 0.5;
      
      console.log(`Spine angle approximated from face: ${spineAngle.toFixed(2)}° (face angle: ${faceAngle.toFixed(2)}°)`);
      return spineAngle;
    }
    
    // 3. If nothing else works, use vertical screen orientation
    console.log("Could not find enough points to calculate spine angle");
    return 0;
  }
  
  isPointVisible(point: PoseKeypoint): boolean {
    if (!point) {
      return false;
    }
    
    // Different models use different confidence thresholds
    // MoveNet uses 'score', BlazePose uses 'visibility'
    const confidence = point.score !== undefined ? point.score : 
                       point.visibility !== undefined ? point.visibility : 0;
    
    const isVisible = confidence > 0.2; // Lower threshold to catch more points
    
    return isVisible;
  }
  
  updateRepCounter(isHinge: boolean): void {
    // Only count a rep when transitioning from hinge to not hinge
    if (this.repCounter.lastHingeState && !isHinge) {
      this.repCounter.count++;
      this.updateRepCounterDisplay();
    }
    
    this.repCounter.lastHingeState = isHinge;
    this.repCounter.isHinge = isHinge;
  }
  
  updateRepCounterDisplay(): void {
    const repCounterElement = document.getElementById('rep-counter');
    if (repCounterElement) {
      repCounterElement.textContent = this.repCounter.count.toString();
    }
  }
  
  updateSpineAngleDisplay(): void {
    const spineAngleElement = document.getElementById('spine-angle');
    if (spineAngleElement) {
      spineAngleElement.textContent = `${Math.round(this.spineAngle)}°`;
    }
  }
  
  drawPose(pose: PoseResult, timestamp: number): void {
    if (!pose.keypoints) return;
    
    // Update UI with keypoint data if callback is provided
    if (this.keypointUpdateCallback && pose.keypoints.length > 0) {
      this.keypointUpdateCallback(pose.keypoints);
    }
    
    const { width, height } = this.canvas;
    
    // Clear canvas with transparent background
    this.ctx.clearRect(0, 0, width, height);
    
    // Calculate FPS
    if (this.startTime === 0) {
      this.startTime = timestamp;
    }
    
    this.frameCount++;
    const elapsed = timestamp - this.startTime;
    if (elapsed >= 1000) {
      this.fps = Math.round((this.frameCount * 1000) / elapsed);
      this.frameCount = 0;
      this.startTime = timestamp;
    }
    
    // Determine what parts of the body are detected
    const hasUpperBody = pose.keypoints.some(kp => 
      (kp.name === 'left_shoulder' || kp.name === 'right_shoulder') && 
      this.isPointVisible(kp)
    );
    
    const hasLowerBody = pose.keypoints.some(kp => 
      (kp.name === 'left_hip' || kp.name === 'right_hip') && 
      this.isPointVisible(kp)
    );
    
    const hasFace = pose.keypoints.some(kp => 
      kp.name === 'nose' && this.isPointVisible(kp)
    );
    
    // Calculate spine angle with whatever points are available
    this.spineAngle = this.calculateSpineVertical(pose.keypoints);
    this.updateSpineAngleDisplay();
    
    // Update rep counter based on spine angle (if we have a valid angle)
    if (this.spineAngle > 0) {
      const isHinge = this.spineAngle < this.repCounter.hingeThreshold;
      this.updateRepCounter(isHinge);
    }
    
    // Draw connections
    this.drawConnections(pose.keypoints);
    
    // Draw keypoints
    this.drawKeypoints(pose.keypoints, timestamp);
    
    // Draw status with semi-transparent background
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    this.ctx.fillRect(5, 5, 300, 120);
    
    this.ctx.fillStyle = 'white';
    this.ctx.font = '16px Arial';
    this.ctx.fillText(`FPS: ${this.fps}`, 10, 25);
    this.ctx.fillText(`Spine Angle: ${Math.round(this.spineAngle)}°`, 10, 50);
    this.ctx.fillText(`Reps: ${this.repCounter.count}`, 10, 75);
    
    // Status message based on what was detected
    if (hasUpperBody && hasLowerBody) {
      this.ctx.fillText(`Status: Full body detected`, 10, 100);
    } else if (hasUpperBody && !hasLowerBody) {
      this.ctx.fillText(`Status: Upper body only (move back)`, 10, 100);
    } else if (!hasUpperBody && hasFace) {
      this.ctx.fillText(`Status: Face only (show more body)`, 10, 100);
    } else {
      this.ctx.fillText(`Status: Limited detection`, 10, 100);
    }
  }
  
  drawConnections(keypoints: PoseKeypoint[]): void {
    // Connection pairs for essential skeleton
    const connections = [
      // Torso
      [MediaPipeBodyParts.LEFT_SHOULDER, MediaPipeBodyParts.RIGHT_SHOULDER],
      [MediaPipeBodyParts.LEFT_SHOULDER, MediaPipeBodyParts.LEFT_HIP],
      [MediaPipeBodyParts.RIGHT_SHOULDER, MediaPipeBodyParts.RIGHT_HIP],
      [MediaPipeBodyParts.LEFT_HIP, MediaPipeBodyParts.RIGHT_HIP],
      
      // Arms
      [MediaPipeBodyParts.LEFT_SHOULDER, MediaPipeBodyParts.LEFT_ELBOW],
      [MediaPipeBodyParts.LEFT_ELBOW, MediaPipeBodyParts.LEFT_WRIST],
      [MediaPipeBodyParts.RIGHT_SHOULDER, MediaPipeBodyParts.RIGHT_ELBOW],
      [MediaPipeBodyParts.RIGHT_ELBOW, MediaPipeBodyParts.RIGHT_WRIST],
      
      // Legs
      [MediaPipeBodyParts.LEFT_HIP, MediaPipeBodyParts.LEFT_KNEE],
      [MediaPipeBodyParts.LEFT_KNEE, MediaPipeBodyParts.LEFT_ANKLE],
      [MediaPipeBodyParts.RIGHT_HIP, MediaPipeBodyParts.RIGHT_KNEE],
      [MediaPipeBodyParts.RIGHT_KNEE, MediaPipeBodyParts.RIGHT_ANKLE],
    ];
    
    // Draw lines between connected keypoints
    this.ctx.lineWidth = 4;
    
    // Get keypoints needed for spine with safety checks
    const leftShoulder = keypoints[MediaPipeBodyParts.LEFT_SHOULDER];
    const rightShoulder = keypoints[MediaPipeBodyParts.RIGHT_SHOULDER];
    const leftHip = keypoints[MediaPipeBodyParts.LEFT_HIP];
    const rightHip = keypoints[MediaPipeBodyParts.RIGHT_HIP];
    
    // Safe array of points that exist and are visible
    const safePoints = [];
    if (leftShoulder && this.isPointVisible(leftShoulder)) safePoints.push(leftShoulder);
    if (rightShoulder && this.isPointVisible(rightShoulder)) safePoints.push(rightShoulder);
    if (leftHip && this.isPointVisible(leftHip)) safePoints.push(leftHip);
    if (rightHip && this.isPointVisible(rightHip)) safePoints.push(rightHip);
    
    // We need at least one shoulder and one hip to draw spine
    const hasAnyShoulder = safePoints.includes(leftShoulder) || safePoints.includes(rightShoulder);
    const hasAnyHip = safePoints.includes(leftHip) || safePoints.includes(rightHip);
    
    if (hasAnyShoulder && hasAnyHip) {
      // Calculate midpoints using only available points
      const shoulderX = safePoints.includes(leftShoulder) && safePoints.includes(rightShoulder) 
        ? (leftShoulder.x + rightShoulder.x) / 2
        : safePoints.includes(leftShoulder) ? leftShoulder.x : rightShoulder.x;
        
      const shoulderY = safePoints.includes(leftShoulder) && safePoints.includes(rightShoulder)
        ? (leftShoulder.y + rightShoulder.y) / 2
        : safePoints.includes(leftShoulder) ? leftShoulder.y : rightShoulder.y;
        
      const hipX = safePoints.includes(leftHip) && safePoints.includes(rightHip)
        ? (leftHip.x + rightHip.x) / 2
        : safePoints.includes(leftHip) ? leftHip.x : rightHip.x;
        
      const hipY = safePoints.includes(leftHip) && safePoints.includes(rightHip)
        ? (leftHip.y + rightHip.y) / 2
        : safePoints.includes(leftHip) ? leftHip.y : rightHip.y;
      
      // Create safe midpoints
      const shoulderMidpoint = { x: shoulderX, y: shoulderY };
      const hipMidpoint = { x: hipX, y: hipY };
      
      // Change color based on angle
      if (this.spineAngle < this.repCounter.hingeThreshold) {
        this.ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)'; // Hinged - semi-transparent red
        this.ctx.lineWidth = 6; // Make spine line thicker
      } else {
        this.ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)'; // Straight - semi-transparent green
        this.ctx.lineWidth = 6; // Make spine line thicker
      }
      
      // Draw spine line
      this.ctx.beginPath();
      this.ctx.moveTo(shoulderMidpoint.x, shoulderMidpoint.y);
      this.ctx.lineTo(hipMidpoint.x, hipMidpoint.y);
      this.ctx.stroke();
    }
    
    // Draw other connections with safety checks
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)'; // Semi-transparent white
    this.ctx.lineWidth = 4;
    for (const [i, j] of connections) {
      const pointA = keypoints[i];
      const pointB = keypoints[j];
      
      // Only draw if both points exist and are visible
      if (!pointA || !pointB || !this.isPointVisible(pointA) || !this.isPointVisible(pointB)) {
        continue;
      }
      
      this.ctx.beginPath();
      this.ctx.moveTo(pointA.x, pointA.y);
      this.ctx.lineTo(pointB.x, pointB.y);
      this.ctx.stroke();
    }
  }
  
  drawKeypoints(keypoints: PoseKeypoint[], timestamp: number): void {
    // Only show body part labels for a set duration from start
    const showLabels = this.showBodyParts && 
      ((timestamp - this.frameTimestamp) / 1000 < this.bodyPartDisplaySeconds);
    
    // Calculate scale factors between canvas intrinsic dimensions and display size
    const canvasElement = this.canvas;
    const scaleX = canvasElement.width / canvasElement.clientWidth;
    const scaleY = canvasElement.height / canvasElement.clientHeight;
    
    // Draw larger, more visible keypoints
    for (let i = 0; i < keypoints.length; i++) {
      const point = keypoints[i];
      if (!this.isPointVisible(point)) continue;
      
      // Transform coordinates if needed to match display size
      const x = point.x;
      const y = point.y;
      
      // Add glow effect
      // Draw outer glow circle
      this.ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      this.ctx.beginPath();
      this.ctx.arc(x, y, 8, 0, 2 * Math.PI);
      this.ctx.fill();
      
      // Draw inner keypoint
      this.ctx.fillStyle = 'rgba(255, 255, 0, 0.8)';
      this.ctx.beginPath();
      this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
      this.ctx.fill();
      
      // Maybe draw the label
      if (showLabels) {
        const partName = this.getBodyPartName(i);
        if (partName) {
          // Draw text background for better readability
          this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
          const textWidth = this.ctx.measureText(partName).width;
          this.ctx.fillRect(x + 10, y - 10, textWidth + 6, 20);
          
          // Draw text
          this.ctx.fillStyle = 'white';
          this.ctx.font = '12px Arial';
          this.ctx.fillText(partName, x + 13, y + 5);
        }
      }
    }
  }
  
  getBodyPartName(index: number): string {
    // Return friendly names for the key body parts
    const names: { [key: number]: string } = {
      [MediaPipeBodyParts.NOSE]: 'Nose',
      [MediaPipeBodyParts.LEFT_SHOULDER]: 'L.Shoulder',
      [MediaPipeBodyParts.RIGHT_SHOULDER]: 'R.Shoulder',
      [MediaPipeBodyParts.LEFT_ELBOW]: 'L.Elbow',
      [MediaPipeBodyParts.RIGHT_ELBOW]: 'R.Elbow',
      [MediaPipeBodyParts.LEFT_WRIST]: 'L.Wrist',
      [MediaPipeBodyParts.RIGHT_WRIST]: 'R.Wrist',
      [MediaPipeBodyParts.LEFT_HIP]: 'L.Hip',
      [MediaPipeBodyParts.RIGHT_HIP]: 'R.Hip',
      [MediaPipeBodyParts.LEFT_KNEE]: 'L.Knee',
      [MediaPipeBodyParts.RIGHT_KNEE]: 'R.Knee',
      [MediaPipeBodyParts.LEFT_ANKLE]: 'L.Ankle',
      [MediaPipeBodyParts.RIGHT_ANKLE]: 'R.Ankle',
    };
    
    return names[index] || '';
  }
  
  async processFrame(timestamp: number): Promise<void> {
    if (this.video.paused || this.video.ended) return;
    
    // Record frame start timestamp if this is the first frame
    if (this.frameTimestamp === 0) {
      this.frameTimestamp = timestamp;
    }
    
    // Limit FPS to improve performance on slower devices
    // Only process every other frame (30fps → 15fps)
    if (this.frameCount % 2 === 0) {
      try {
        const pose = await this.detectPose();
        if (pose && pose.keypoints) {
          this.drawPose(pose, timestamp);
        } else {
          // If no pose detected, just update the status text
          // This avoids freezing the UI when no pose is detected
          const ctx = this.canvas.getContext('2d');
          if (ctx) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.fillRect(10, 10, 300, 30);
            ctx.fillStyle = 'white';
            ctx.font = '16px Arial';
            ctx.fillText('No pose detected - adjust position or lighting', 20, 30);
          }
        }
      } catch (error) {
        console.error('Error in frame processing:', error);
      }
    }
    
    this.frameCount++;
    
    // Continue processing frames
    this.rafId = requestAnimationFrame(this.processFrame.bind(this));
  }
  
  startProcessing(): void {
    if (this.rafId) return;
    this.rafId = requestAnimationFrame(this.processFrame.bind(this));
  }
  
  stopProcessing(): void {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }
  
  reset(): void {
    this.repCounter.count = 0;
    this.repCounter.isHinge = false;
    this.repCounter.lastHingeState = false;
    this.frameTimestamp = 0;
    this.updateRepCounterDisplay();
  }
  
  setBodyPartDisplay(show: boolean, seconds: number): void {
    this.showBodyParts = show;
    this.bodyPartDisplaySeconds = seconds;
  }
}
