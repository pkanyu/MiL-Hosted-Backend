#!/usr/bin/env python3
"""
VideoVeritas ULTIMATE Detection System - WITH AI IMAGE DETECTION
Now includes comprehensive AI image analysis for single frames
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import tempfile
from datetime import datetime
import json
import base64
from scipy import ndimage
from skimage import feature
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
# More permissive CORS for Brave browser
CORS(app, origins="*", allow_headers="*", methods=["GET", "POST", "OPTIONS"])

class UltimateVideoVeritas:
    """
    Enhanced detector with AI image detection capabilities
    """
    
    def __init__(self):
        self.frame_sample_rate = 30
        self.suspicious_indicators = []
        self.ai_signatures_found = []
        
        # Calibrated thresholds
        self.thresholds = {
            'dct_uniformity': 1.0,
            'noise_clean': 0.3,
            'noise_excessive': 20,
            'motion_consistent': 1.5,
            'motion_erratic': 18,
            'block_variance': 30,
            'edge_stability_low': 0.25,
            'edge_stability_high': 0.97,
        }
        
    # ========== NEW AI IMAGE DETECTION METHODS ==========
    
    def detect_gan_fingerprints(self, image):
        """Detect GAN-specific artifacts in images"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = 0
        
        # Check for GAN mode collapse patterns
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # GANs often produce specific histogram patterns
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        if entropy < 6.5:  # Unnaturally low entropy
            score += 40
            self.ai_signatures_found.append("GAN histogram anomaly detected")
        
        # Check for checkerboard artifacts (common in deconvolution)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Look for regular grid patterns in frequency domain
        h, w = magnitude.shape
        center_region = magnitude[h//4:3*h//4, w//4:3*w//4]
        peaks = feature.peak_local_max(center_region, min_distance=5)
        
        if len(peaks) > 10:  # Multiple regular peaks indicate artifacts
            score += 30
            self.ai_signatures_found.append("Checkerboard artifacts detected")
            
        return score
    
    def detect_diffusion_artifacts(self, image):
        """Detect diffusion model specific artifacts"""
        score = 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Diffusion models often leave specific noise patterns
        # Analyze high-frequency residuals
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residual = gray.astype(float) - blurred.astype(float)
        
        # Check residual statistics
        residual_std = np.std(residual)
        residual_mean = np.abs(np.mean(residual))
        
        # Diffusion models have characteristic residual patterns
        if residual_std < 3.0 and residual_mean < 0.5:
            score += 35
            self.suspicious_indicators.append("Diffusion model noise signature")
        
        # Check for over-smoothing (common in DDPM/DDIM)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        if edge_density < 0.02:  # Too few edges
            score += 25
            self.suspicious_indicators.append("Over-smoothed surfaces (diffusion)")
            
        return score
    
    def detect_stylegan_signatures(self, image):
        """Detect StyleGAN specific artifacts"""
        score = 0
        
        # StyleGAN often produces specific color distributions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze saturation channel
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        # StyleGAN tends to produce oversaturated images
        if sat_mean > 140 and sat_std < 40:
            score += 30
            self.ai_signatures_found.append("StyleGAN saturation pattern")
        
        # Check for face-specific artifacts if face detected
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_region = image[y:y+h, x:x+w]
                
                # Check for unnatural skin texture
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                texture_variance = np.var(face_gray)
                
                if texture_variance < 200:  # Too smooth
                    score += 40
                    self.ai_signatures_found.append("Unnatural skin texture (StyleGAN)")
                    
        return score
    
    def detect_unnatural_textures(self, image):
        """Detect textures that are too perfect or uniform"""
        score = 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute Local Binary Patterns for texture analysis
        def compute_lbp(img, radius=1, n_points=8):
            rows, cols = img.shape
            lbp = np.zeros_like(img)
            
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = img[i, j]
                    binary_string = ''
                    
                    for n in range(n_points):
                        theta = 2 * np.pi * n / n_points
                        x = int(round(i + radius * np.cos(theta)))
                        y = int(round(j + radius * np.sin(theta)))
                        
                        if img[x, y] >= center:
                            binary_string += '1'
                        else:
                            binary_string += '0'
                    
                    lbp[i, j] = int(binary_string, 2)
            
            return lbp
        
        # Compute LBP
        lbp = compute_lbp(gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        lbp_hist = lbp_hist.flatten() / lbp_hist.sum()
        
        # AI images often have less texture variety
        texture_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        
        if texture_entropy < 5.0:
            score += 35
            self.suspicious_indicators.append("Unnaturally uniform textures")
            
        return score
    
    def detect_color_banding(self, image):
        """Detect color banding artifacts common in AI images"""
        score = 0
        
        # Check each color channel for banding
        for channel in range(3):
            chan = image[:, :, channel]
            
            # Compute gradient
            grad_x = cv2.Sobel(chan, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(chan, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Look for sudden jumps (banding)
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Count gaps in histogram (indicates banding)
            gaps = 0
            for i in range(1, 255):
                if hist[i] == 0 and hist[i-1] > 0 and hist[i+1] > 0:
                    gaps += 1
            
            if gaps > 20:
                score += 20
                if "Color banding detected" not in self.suspicious_indicators:
                    self.suspicious_indicators.append("Color banding detected")
                    
        return score
    
    def detect_ai_sharpness_patterns(self, image):
        """Detect unnatural sharpness patterns typical of AI images"""
        score = 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute Laplacian to measure sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # AI images often have specific sharpness ranges
        if 100 < sharpness < 500:  # Suspiciously consistent sharpness
            score += 25
            self.suspicious_indicators.append("AI-typical sharpness pattern")
        
        # Check for over-sharpening artifacts
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        diff = np.abs(sharpened.astype(float) - gray.astype(float))
        
        if np.mean(diff) < 5:  # Image already appears over-sharpened
            score += 20
            self.ai_signatures_found.append("Pre-sharpened appearance")
            
        return score
    
    def detect_latent_space_artifacts(self, image):
        """Detect artifacts from latent space interpolation"""
        score = 0
        
        # Check for impossible object boundaries
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                # Check contour smoothness
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Unnatural perfect shapes
                    if 0.95 < circularity < 1.0:
                        score += 15
                        if "Unnaturally perfect shapes" not in self.ai_signatures_found:
                            self.ai_signatures_found.append("Unnaturally perfect shapes")
                            
        return score
    
    def detect_model_specific_signatures(self, image):
        """Detect signatures specific to popular AI models"""
        score = 0
        
        # DALL-E tends to add vignetting
        h, w = image.shape[:2]
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        border_region = np.concatenate([
            image[:h//4, :].flatten(),
            image[3*h//4:, :].flatten(),
            image[:, :w//4].flatten(),
            image[:, 3*w//4:].flatten()
        ])
        
        center_brightness = np.mean(center_region)
        border_brightness = np.mean(border_region)
        
        if center_brightness > border_brightness * 1.2:
            score += 20
            self.ai_signatures_found.append("DALL-E style vignetting")
        
        # Midjourney often has specific color grading
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hue_hist = hue_hist.flatten() / hue_hist.sum()
        
        # Check for characteristic peaks
        peaks = feature.peak_local_max(hue_hist.reshape(-1, 1), min_distance=10)
        if len(peaks) == 2 or len(peaks) == 3:  # Characteristic dual/triple tone
            score += 25
            self.ai_signatures_found.append("Midjourney color grading pattern")
            
        return score
    
    def analyze_single_frame(self, frame):
        """Enhanced single frame analysis with AI image detection"""
        frames = [frame]
        self.suspicious_indicators = []
        self.ai_signatures_found = []
        
        # Original video-based single frame analysis
        video_scores = {
            'dct': self.analyze_dct_calibrated(frames),
            'noise': self.analyze_noise_calibrated(frames),
            'cleanliness': self.analyze_cleanliness_v1(frames),
            'frequency': self.analyze_frequency_calibrated(frames),
            'compression': self.detect_compression_artifacts(frames),
            'deepfake': self.detect_deepfake_signatures(frames),
        }
        
        # NEW: AI image detection scores
        ai_image_scores = {
            'gan_fingerprints': self.detect_gan_fingerprints(frame),
            'diffusion_artifacts': self.detect_diffusion_artifacts(frame),
            'stylegan_signatures': self.detect_stylegan_signatures(frame),
            'unnatural_textures': self.detect_unnatural_textures(frame),
            'color_banding': self.detect_color_banding(frame),
            'ai_sharpness': self.detect_ai_sharpness_patterns(frame),
            'latent_space': self.detect_latent_space_artifacts(frame),
            'model_signatures': self.detect_model_specific_signatures(frame)
        }
        
        # Count how many AI detectors triggered
        ai_triggers = sum(1 for score in ai_image_scores.values() if score > 20)
        video_triggers = sum(1 for score in video_scores.values() if score > 30)
        
        # Apply boost if multiple detectors triggered
        confidence_boost = 0
        if ai_triggers >= 3:
            confidence_boost = 25
            self.ai_signatures_found.append("Multiple AI patterns detected")
        elif ai_triggers >= 2:
            confidence_boost = 15
        elif ai_triggers >= 1:
            confidence_boost = 5
            
        # Combine scores with calibrated weights
        combined_weights = {
            # Video-based (reduced weight for single frame)
            'dct': 0.03,
            'noise': 0.03,
            'cleanliness': 0.03,
            'frequency': 0.03,
            'compression': 0.03,
            'deepfake': 0.05,
            # AI image-based (increased weight)
            'gan_fingerprints': 0.15,
            'diffusion_artifacts': 0.12,
            'stylegan_signatures': 0.12,
            'unnatural_textures': 0.10,
            'color_banding': 0.08,
            'ai_sharpness': 0.10,
            'latent_space': 0.06,
            'model_signatures': 0.07
        }
        
        all_scores = {**video_scores, **ai_image_scores}
        weighted_score = sum(all_scores[key] * combined_weights[key] for key in all_scores)
        
        # Apply confidence boost
        weighted_score += confidence_boost
        
        # Apply scaling to make scores more decisive
        # Scores below 20 -> scale down (more likely authentic)
        # Scores above 20 -> scale up (more likely AI)
        if weighted_score < 20:
            final_score = weighted_score * 0.5  # Scale down for authentic
        elif weighted_score < 40:
            final_score = weighted_score * 1.5  # Scale up for uncertain
        else:
            final_score = min(weighted_score * 1.8, 95)  # Scale up for AI
        
        # Ensure minimum detection for any triggers
        if ai_triggers > 0 and final_score < 25:
            final_score = 25
        
        # Determine detected model based on signatures
        detected_model = self.identify_ai_model_from_image(ai_image_scores)
        
        # Remove duplicates
        self.suspicious_indicators = list(dict.fromkeys(self.suspicious_indicators))
        self.ai_signatures_found = list(dict.fromkeys(self.ai_signatures_found))
        
        return {
            "ai_likelihood_percent": round(final_score, 1),
            "confidence_level": self.get_confidence_level(final_score),
            "suspicious_indicators": self.suspicious_indicators if self.suspicious_indicators else ["No significant AI artifacts detected"],
            "ai_signatures": self.ai_signatures_found,
            "detailed_scores": {
                **{k: round(v, 1) for k, v in video_scores.items()},
                **{k: round(v, 1) for k, v in ai_image_scores.items()}
            },
            "detected_model": detected_model,
            "analysis_type": "AI Image Detection (Single Frame)"
        }
    
    def identify_ai_model_from_image(self, scores):
        """Identify AI model based on image analysis scores"""
        
        if scores['stylegan_signatures'] > 40:
            return "StyleGAN/ThisPersonDoesNotExist"
        elif scores['diffusion_artifacts'] > 35:
            return "Stable Diffusion/DALL-E"
        elif scores['gan_fingerprints'] > 40:
            return "GAN-based Model"
        elif scores['model_signatures'] > 30:
            return "Midjourney/DALL-E"
        elif max(scores.values()) > 40:
            return "Unknown AI Model (High Confidence)"
        elif max(scores.values()) > 25:
            return "Possible AI Generation"
        else:
            return "Likely Authentic"
    
    # ========== ORIGINAL VIDEO ANALYSIS METHODS (kept for full video) ==========
    
    def analyze_video(self, video_path):
        """Main analysis for full videos"""
        self.suspicious_indicators = []
        self.ai_signatures_found = []
        
        frames = self.extract_frames(video_path)
        if not frames:
            return {"error": "Could not process video"}
        
        scores = {}
        
        scores['motion'] = self.analyze_motion_patterns_v1(frames)
        scores['lighting'] = self.analyze_lighting_consistency_v1(frames)
        scores['temporal'] = self.analyze_temporal_artifacts_v1(frames)
        scores['cleanliness'] = self.analyze_cleanliness_v1(frames)
        scores['dct'] = self.analyze_dct_calibrated(frames)
        scores['noise'] = self.analyze_noise_calibrated(frames)
        scores['edges'] = self.analyze_edge_coherence_calibrated(frames)
        scores['frequency'] = self.analyze_frequency_calibrated(frames)
        scores['compression'] = self.detect_compression_artifacts(frames)
        scores['deepfake'] = self.detect_deepfake_signatures(frames)
        
        ai_likelihood = self.calculate_smart_score(scores, frames)
        detected_model = self.identify_ai_model_smart(scores)
        
        self.suspicious_indicators = list(dict.fromkeys(self.suspicious_indicators))
        
        return {
            "ai_likelihood_percent": round(ai_likelihood, 1),
            "confidence_level": self.get_confidence_level(ai_likelihood),
            "detected_model": detected_model,
            "suspicious_indicators": self.suspicious_indicators,
            "ai_signatures": self.ai_signatures_found,
            "detection_scores": {
                "motion_analysis": round(scores['motion'], 1),
                "lighting_consistency": round(scores['lighting'], 1),
                "temporal_stability": round(scores['temporal'], 1),
                "image_quality": round(scores['cleanliness'], 1),
                "dct_patterns": round(scores['dct'], 1),
                "noise_analysis": round(scores['noise'], 1),
                "edge_coherence": round(scores['edges'], 1),
                "frequency_analysis": round(scores['frequency'], 1),
                "compression_artifacts": round(scores['compression'], 1),
                "deepfake_detection": round(scores['deepfake'], 1)
            }
        }
    
    def extract_frames(self, video_path):
        """Extract frames intelligently"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 100:
            sample_rate = 5
        elif total_frames < 500:
            sample_rate = 15
        else:
            sample_rate = 30
            
        frame_count = 0
        while len(frames) < 30:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def analyze_motion_patterns_v1(self, frames):
        """Motion detection"""
        if len(frames) < 3:
            return 0
        
        diff_scores = []
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray1, gray2)
            diff_score = np.mean(diff)
            diff_scores.append(diff_score)
        
        if diff_scores:
            avg_diff = np.mean(diff_scores)
            std_diff = np.std(diff_scores)
            
            if std_diff < self.thresholds['motion_consistent']:
                self.suspicious_indicators.append("Unnaturally consistent motion")
                return 75
            elif std_diff > self.thresholds['motion_erratic']:
                self.suspicious_indicators.append("Erratic motion patterns")
                return 65
            elif avg_diff < 3.0:
                self.suspicious_indicators.append("Minimal motion variation")
                return 70
        
        return 25
    
    def analyze_lighting_consistency_v1(self, frames):
        """Lighting analysis"""
        if len(frames) < 2:
            return 0
        
        lighting_scores = []
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            lighting_scores.append(correlation)
        
        avg_consistency = np.mean(lighting_scores)
        
        if avg_consistency < 0.7:
            self.suspicious_indicators.append("Inconsistent lighting patterns")
        
        suspicion_score = max(0, (0.9 - avg_consistency) * 100)
        return min(suspicion_score, 100)
    
    def analyze_temporal_artifacts_v1(self, frames):
        """Temporal analysis"""
        if len(frames) < 3:
            return 0
        
        dramatic_changes = 0
        for i in range(len(frames) - 1):
            frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            mean_diff = abs(np.mean(frame1) - np.mean(frame2))
            
            if mean_diff > 80:
                dramatic_changes += 1
        
        if dramatic_changes > len(frames) * 0.3:
            self.suspicious_indicators.append("Extreme temporal inconsistencies")
            return min(dramatic_changes * 25, 60)
        
        return 0
    
    def analyze_cleanliness_v1(self, frames):
        """Cleanliness detector"""
        if not frames:
            return 0
        
        cleanliness_scores = []
        for frame in frames[:3]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = laplacian.var()
            
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / edges.size
            
            if noise_level < 100 and edge_density > 0.05:
                cleanliness_scores.append(1)
            else:
                cleanliness_scores.append(0)
        
        if cleanliness_scores:
            cleanliness_ratio = np.mean(cleanliness_scores)
            if cleanliness_ratio > 0.6:
                self.suspicious_indicators.append("Unnaturally clean image")
                return min(cleanliness_ratio * 60, 30)
        
        return 0
    
    def analyze_dct_calibrated(self, frames):
        """DCT analysis"""
        dct_scores = []
        
        for frame in frames[:5]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            dct_coeffs = []
            
            for y in range(0, h-8, 8):
                for x in range(0, w-8, 8):
                    block = gray[y:y+8, x:x+8].astype(np.float32)
                    dct = cv2.dct(block)
                    high_freq = np.abs(dct[4:, 4:]).mean()
                    dct_coeffs.append(high_freq)
            
            if dct_coeffs:
                coeff_std = np.std(dct_coeffs)
                
                if coeff_std < self.thresholds['dct_uniformity']:
                    dct_scores.append(60)
                    if "Uniform DCT coefficients" not in self.suspicious_indicators:
                        self.suspicious_indicators.append("Uniform DCT coefficients")
                elif coeff_std < 3.0:
                    dct_scores.append(30)
                else:
                    dct_scores.append(5)
        
        return np.mean(dct_scores) if dct_scores else 0
    
    def analyze_noise_calibrated(self, frames):
        """Noise analysis"""
        noise_scores = []
        
        for frame in frames[:5]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            blur1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
            blur2 = cv2.GaussianBlur(gray, (5, 5), 2.0)
            
            noise = cv2.absdiff(blur1, blur2)
            noise_level = np.std(noise)
            
            if noise_level < self.thresholds['noise_clean']:
                noise_scores.append(50)
                if "Unnaturally clean" not in self.suspicious_indicators:
                    self.suspicious_indicators.append("Unnaturally clean (no noise)")
            elif noise_level > self.thresholds['noise_excessive']:
                noise_scores.append(40)
            else:
                noise_scores.append(10)
        
        return np.mean(noise_scores) if noise_scores else 0
    
    def analyze_edge_coherence_calibrated(self, frames):
        """Edge analysis"""
        if len(frames) < 2:
            return 0
        
        edge_scores = []
        
        for i in range(min(len(frames) - 1, 10)):
            edges1 = cv2.Canny(frames[i], 50, 150)
            edges2 = cv2.Canny(frames[i + 1], 50, 150)
            
            stable_edges = cv2.bitwise_and(edges1, edges2)
            
            if np.sum(edges1) > 0:
                stability_ratio = np.sum(stable_edges) / np.sum(edges1)
                
                if stability_ratio < self.thresholds['edge_stability_low']:
                    edge_scores.append(50)
                    if "Edge instability" not in self.suspicious_indicators:
                        self.suspicious_indicators.append("Edge instability between frames")
                elif stability_ratio > self.thresholds['edge_stability_high']:
                    edge_scores.append(40)
                else:
                    edge_scores.append(10)
        
        return np.mean(edge_scores) if edge_scores else 0
    
    def analyze_frequency_calibrated(self, frames):
        """Frequency analysis"""
        freq_scores = []
        
        for frame in frames[:3]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            
            ring_energies = []
            for radius in range(10, min(center_h, center_w), 20):
                y, x = np.ogrid[:h, :w]
                mask = ((x - center_w)**2 + (y - center_h)**2 >= radius**2) & \
                       ((x - center_w)**2 + (y - center_h)**2 < (radius + 20)**2)
                ring_energy = np.mean(magnitude[mask])
                ring_energies.append(ring_energy)
            
            if ring_energies:
                energy_std = np.std(ring_energies)
                
                if energy_std < 80:
                    freq_scores.append(40)
                    if "Artificial frequency distribution" not in self.ai_signatures_found:
                        self.ai_signatures_found.append("Artificial frequency distribution")
                else:
                    freq_scores.append(10)
        
        return np.mean(freq_scores) if freq_scores else 0
    
    def detect_compression_artifacts(self, frames):
        """Compression artifact detection"""
        compression_scores = []
        
        for frame in frames[:5]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            h, w = gray.shape
            block_strength_8 = 0
            block_strength_16 = 0
            
            for x in range(8, w, 8):
                block_strength_8 += np.mean(np.abs(sobelx[:, x-1:x+1]))
            for y in range(8, h, 8):
                block_strength_8 += np.mean(np.abs(sobely[y-1:y+1, :]))
                
            for x in range(16, w, 16):
                block_strength_16 += np.mean(np.abs(sobelx[:, x-1:x+1]))
            for y in range(16, h, 16):
                block_strength_16 += np.mean(np.abs(sobely[y-1:y+1, :]))
            
            if block_strength_8 > block_strength_16 * 1.5:
                compression_scores.append(40)
            elif block_strength_16 > block_strength_8 * 1.5:
                compression_scores.append(35)
                if "16x16 block compression" not in self.ai_signatures_found:
                    self.ai_signatures_found.append("16x16 block compression pattern")
            else:
                compression_scores.append(10)
        
        return np.mean(compression_scores) if compression_scores else 0
    
    def detect_deepfake_signatures(self, frames):
        """Deepfake detection"""
        deepfake_scores = []
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        for frame in frames[:5]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_region = gray[y:y+h, x:x+w]
                    
                    if w > 20:
                        left_half = face_region[:, :w//2]
                        right_half = cv2.flip(face_region[:, w//2:], 1)
                        
                        if left_half.shape == right_half.shape:
                            correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                            
                            if correlation > 0.92:
                                deepfake_scores.append(60)
                                if "Unnatural facial symmetry" not in self.ai_signatures_found:
                                    self.ai_signatures_found.append("Unnatural facial symmetry detected")
                            else:
                                deepfake_scores.append(15)
                    
                    eye_region = face_region[h//4:h//2, :]
                    eye_variance = np.var(eye_region)
                    
                    if eye_variance < 80:
                        deepfake_scores.append(40)
                        if "Eye region artifacts" not in self.suspicious_indicators:
                            self.suspicious_indicators.append("Eye region artifacts")
        
        return np.mean(deepfake_scores) if deepfake_scores else 0
    
    def calculate_smart_score(self, scores, frames):
        """Smart scoring system"""
        triggered = sum(1 for s in scores.values() if s > 30)
        
        weights = {
            'motion': 0.25,
            'lighting': 0.10,
            'temporal': 0.10,
            'cleanliness': 0.08,
            'dct': 0.12,
            'noise': 0.08,
            'edges': 0.10,
            'frequency': 0.07,
            'compression': 0.05,
            'deepfake': 0.05
        }
        
        if triggered >= 5:
            confidence_multiplier = 1.15
        elif triggered >= 3:
            confidence_multiplier = 1.05
        else:
            confidence_multiplier = 1.0
        
        weighted_score = sum(scores[key] * weights[key] for key in scores)
        final_score = min(weighted_score * confidence_multiplier, 100)
        
        if scores['motion'] > 60 and scores['edges'] > 40 and scores['dct'] > 40:
            final_score = max(final_score, 65)
            self.ai_signatures_found.append("Multiple AI patterns detected")
        
        if all(s < 20 for s in scores.values()):
            final_score = min(final_score, 15)
        
        return final_score
    
    def identify_ai_model_smart(self, scores):
        """Model identification"""
        if scores['compression'] > 35 and scores['frequency'] > 35:
            self.ai_signatures_found.append("VeO-like compression signature")
            return "VeO (probable)"
        
        elif scores['noise'] > 40 and scores['cleanliness'] > 25:
            self.ai_signatures_found.append("Sora-like quality patterns")
            return "Sora (probable)"
        
        elif scores['temporal'] > 40 or scores['edges'] > 45:
            self.ai_signatures_found.append("Runway-like temporal patterns")
            return "Runway (probable)"
        
        elif scores['deepfake'] > 40:
            return "Deepfake Generator"
        
        elif max(scores.values()) > 50:
            return "Unknown AI Model"
        
        else:
            return "Likely Authentic"
    
    def get_confidence_level(self, score):
        """Confidence level"""
        if score >= 65:
            return "🔴 DEFINITELY AI-Generated"
        elif score >= 45:
            return "🟠 LIKELY AI-Generated"
        elif score >= 30:
            return "🟡 POSSIBLY AI-Generated"
        elif score >= 20:
            return "🔵 UNLIKELY AI-Generated"
        else:
            return "🟢 AUTHENTIC"

# Initialize detector
detector = UltimateVideoVeritas()

# ============= FLASK ROUTES =============

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VideoVeritas Ultimate</title>
        <style>
            body { font-family: Arial; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { text-align: center; }
            .status { background: rgba(0,255,0,0.2); padding: 10px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 VideoVeritas ULTIMATE Detection</h1>
            <div class="status">
                ✅ System Online - With AI Image Detection<br>
                ✅ Enhanced single-frame accuracy<br>
                ✅ Detects GAN, Diffusion, StyleGAN artifacts<br>
                ✅ Model-specific signature detection
            </div>
            <p>Single frames now analyzed as AI images for better accuracy!</p>
        </div>
    </body>
    </html>
    """

@app.route('/api/test', methods=['GET', 'OPTIONS'])
def test_connection():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
    
    return jsonify({
        'status': 'success',
        'message': 'VideoVeritas Ultimate is running',
        'version': '4.0-ai-image-detection'
    })

@app.route('/api/analyze-frame', methods=['POST', 'OPTIONS'])
def analyze_frame():
    """Single frame analysis with AI image detection"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
    
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        frame_file = request.files['frame']
        frame_data = frame_file.read()
        
        print(f"Received frame data: {len(frame_data)} bytes")
        
        if len(frame_data) == 0:
            return jsonify({'error': 'Empty frame data received'}), 400
        
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("First decode attempt failed, trying IMREAD_UNCHANGED")
            frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if frame is None:
            print("Second decode attempt failed, trying IMREAD_GRAYSCALE")
            frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        if frame is None:
            try:
                decoded_data = base64.b64decode(frame_data)
                nparr = np.frombuffer(decoded_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except:
                pass
            
            if frame is None:
                return jsonify({'error': 'Could not decode frame - invalid image data'}), 400
        
        print(f"Frame decoded successfully: {frame.shape}")
        print("Running AI image detection on single frame...")
        
        result = detector.analyze_single_frame(frame)
        
        print(f"Analysis complete - AI likelihood: {result['ai_likelihood_percent']}%")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_frame: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-advanced', methods=['POST'])
def analyze_advanced():
    """Full video analysis"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        temp_path = os.path.join(tempfile.gettempdir(), f'video_{datetime.now().timestamp()}.mp4')
        file.save(temp_path)
        
        try:
            result = detector.analyze_video(temp_path)
            return jsonify(result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For production
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)