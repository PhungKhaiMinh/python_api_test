#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ensemble HOG + ORB Classifier cho HMI Screen Detection
Kết hợp HOG + SVM với ORB features để đạt >90% accuracy
với tốc độ xử lý <10 giây

Features:
- HOG + SVM classifier (base)
- ORB features extraction và matching
- Weighted ensemble voting
- Confidence optimization
- Advanced data augmentation

Author: AI Assistant
Date: 2024
"""

import os
import cv2
import numpy as np
import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EnsembleResult:
    """Kết quả ensemble classification"""
    predicted_screen: str
    confidence: float
    processing_time: float
    hog_prediction: str
    hog_confidence: float
    orb_prediction: str
    orb_confidence: float
    ensemble_weights: Dict[str, float]

class EnsembleHOGORBClassifier:
    """
    Ensemble Classifier kết hợp HOG + SVM với ORB features
    Tối ưu để đạt >90% accuracy với tốc độ <10s
    """
    
    def __init__(self, model_save_path: str = "ensemble_hog_orb_model.pkl"):
        self.model_save_path = model_save_path
        
        # HOG + SVM components
        self.hog_svm_model = None
        self.hog_scaler = None
        self.hog_label_encoder = None
        
        # ORB components
        self.orb_classifier = None
        self.orb_scaler = None
        self.orb_label_encoder = None
        self.orb_reference_features = {}  # Store reference ORB features
        
        # Ensemble components
        self.ensemble_weights = {
            'hog': 0.6,  # HOG có weight cao hơn do đã proven
            'orb': 0.4   # ORB bổ sung cho edge cases
        }
        self.confidence_threshold = 0.15  # Further lowered for better recall
        
        # Feature extraction parameters
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'transform_sqrt': True,
            'feature_vector': True
        }
        self.image_size = (128, 128)
        self.orb = cv2.ORB_create(nfeatures=500)  # More features for better matching
        
        self.is_trained = False
        self._load_model()
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing với noise reduction"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = rgb2gray(image)
            else:
                gray = image
            
            # Resize to standard size
            resized = resize(gray, self.image_size, anti_aliasing=True)
            
            # Normalize intensity
            resized = (resized * 255).astype(np.uint8)
            
            # Advanced preprocessing pipeline
            # 1. Histogram equalization
            equalized = cv2.equalizeHist(resized)
            
            # 2. Noise reduction
            denoised = cv2.bilateralFilter(equalized, 9, 75, 75)
            
            # 3. Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            return enhanced / 255.0  # Normalize back to [0,1]
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
    
    def _extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features"""
        try:
            processed_image = self._preprocess_image(image)
            if processed_image is None:
                return None
            
            features = hog(
                processed_image,
                orientations=self.hog_params['orientations'],
                pixels_per_cell=self.hog_params['pixels_per_cell'],
                cells_per_block=self.hog_params['cells_per_block'],
                block_norm=self.hog_params['block_norm'],
                transform_sqrt=self.hog_params['transform_sqrt'],
                feature_vector=self.hog_params['feature_vector']
            )
            
            return features
            
        except Exception as e:
            print(f"Error extracting HOG features: {e}")
            return None
    
    def _extract_orb_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ORB keypoints và descriptors"""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            if processed_image is None:
                return None, None
            
            # Convert to uint8 for ORB
            processed_uint8 = (processed_image * 255).astype(np.uint8)
            
            # Extract ORB features
            keypoints, descriptors = self.orb.detectAndCompute(processed_uint8, None)
            
            if descriptors is None or len(descriptors) < 10:
                return None, None
            
            return keypoints, descriptors
            
        except Exception as e:
            print(f"Error extracting ORB features: {e}")
            return None, None
    
    def _extract_combined_features(self, image: np.ndarray) -> Dict:
        """Extract cả HOG và ORB features"""
        features = {}
        
        # HOG features
        hog_features = self._extract_hog_features(image)
        features['hog'] = hog_features
        
        # ORB features
        keypoints, descriptors = self._extract_orb_features(image)
        features['orb_keypoints'] = keypoints
        features['orb_descriptors'] = descriptors
        
        # Additional features
        features['color_hist'] = self._extract_color_histogram(image)
        features['edge_density'] = self._extract_edge_density(image)
        
        return features
    
    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        try:
            if len(image.shape) == 3:
                # Convert BGR to HSV
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # Calculate histogram for H and S channels
                hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
                hist = np.concatenate([hist_h.flatten(), hist_s.flatten()])
            else:
                # Grayscale histogram
                hist = cv2.calcHist([image], [0], None, [64], [0, 256])
                hist = hist.flatten()
            
            # Normalize
            hist = hist / (np.sum(hist) + 1e-7)
            return hist
            
        except Exception as e:
            print(f"Error extracting color histogram: {e}")
            return np.zeros(64)
    
    def _extract_edge_density(self, image: np.ndarray) -> float:
        """Extract edge density feature"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return edge_density
            
        except Exception as e:
            print(f"Error extracting edge density: {e}")
            return 0.0
    
    def _match_orb_features(self, test_descriptors: np.ndarray, reference_descriptors: np.ndarray) -> float:
        """Match ORB features và return similarity score"""
        try:
            if test_descriptors is None or reference_descriptors is None:
                return 0.0
            
            # Use BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(test_descriptors, reference_descriptors)
            
            if len(matches) == 0:
                return 0.0
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate match score
            good_matches = [m for m in matches if m.distance < 50]
            match_ratio = len(good_matches) / min(len(test_descriptors), len(reference_descriptors))
            
            # Weight by match quality
            avg_distance = np.mean([m.distance for m in good_matches]) if good_matches else 100
            quality_score = max(0, 1 - avg_distance / 100)
            
            final_score = match_ratio * quality_score
            return min(1.0, final_score)
            
        except Exception as e:
            print(f"Error in ORB matching: {e}")
            return 0.0
    
    def _extract_screen_from_filename(self, filename: str) -> str:
        """Extract screen type từ filename"""
        try:
            if filename.startswith('template_'):
                name_part = filename.replace('.jpg', '').replace('.png', '').replace('template_', '')
                parts = name_part.split('_', 1)
                
                if len(parts) >= 2:
                    screen_type = parts[1]
                    base_screen = screen_type.split('_')[0]
                    
                    screen_mapping = {
                        'Temp': 'Temperature',
                        'Setting': 'Setup',
                        'Main': 'Main',
                        'Feeders': 'Feeder',
                        'Production': 'Production',
                        'Data': 'Data',
                        'Selectors': 'Maintenance'
                    }
                    
                    return screen_mapping.get(base_screen, base_screen)
            
            # Fallback logic
            filename_lower = filename.lower()
            for screen_type in ['production', 'temperature', 'overview', 'plasticizer', 'setup', 'tracking']:
                if screen_type in filename_lower:
                    return screen_type.capitalize()
            
            return 'Unknown'
            
        except Exception as e:
            print(f"Error extracting screen from filename {filename}: {e}")
            return 'Unknown'
    
    def train_from_folders(self, folder_paths: List[str], test_size: float = 0.1) -> Dict:
        """Train ensemble model từ các folder"""
        print("🚀 Starting Ensemble HOG + ORB Training")
        start_time = time.time()
        
        # Collect training data
        images = []
        labels = []
        
        for folder_path in folder_paths:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                continue
            
            print(f"📁 Processing folder: {folder_path}")
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
                image_files.extend(folder_path.glob(f"*{ext.upper()}"))
            
            for image_file in image_files:
                try:
                    image = cv2.imread(str(image_file))
                    if image is None:
                        continue
                    
                    screen_type = self._extract_screen_from_filename(image_file.name)
                    if screen_type == 'Unknown':
                        continue
                    
                    images.append(image)
                    labels.append(screen_type)
                    
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    continue
        
        if len(images) == 0:
            return {'error': 'No valid training images'}
        
        print(f"📊 Total training images: {len(images)}")
        unique_labels = list(set(labels))
        print(f"📋 Screen types: {unique_labels}")
        
        # Extract features for all images
        print("🔄 Extracting combined features...")
        hog_features = []
        orb_reference_features = {}
        color_features = []
        edge_features = []
        valid_labels = []
        
        for i, (image, label) in enumerate(zip(images, labels)):
            if i % 10 == 0:
                print(f"Progress: {i+1}/{len(images)}")
            
            features = self._extract_combined_features(image)
            
            if features['hog'] is not None:
                hog_features.append(features['hog'])
                color_features.append(features['color_hist'])
                edge_features.append([features['edge_density']])
                valid_labels.append(label)
                
                # Store ORB reference features
                if features['orb_descriptors'] is not None:
                    if label not in orb_reference_features:
                        orb_reference_features[label] = []
                    orb_reference_features[label].append(features['orb_descriptors'])
        
        if len(hog_features) == 0:
            return {'error': 'No valid features extracted'}
        
        # Train HOG + SVM
        print("🔄 Training HOG + SVM...")
        hog_features = np.array(hog_features)
        color_features = np.array(color_features)
        edge_features = np.array(edge_features)
        
        # Combine HOG with additional features
        combined_features = np.hstack([hog_features, color_features, edge_features])
        
        self.hog_label_encoder = LabelEncoder()
        encoded_labels = self.hog_label_encoder.fit_transform(valid_labels)
        
        # Scale features
        self.hog_scaler = StandardScaler()
        
        if len(unique_labels) <= len(combined_features) * (1 - test_size):
            X_train, X_test, y_train, y_test = train_test_split(
                combined_features, encoded_labels, test_size=test_size, 
                random_state=42, stratify=encoded_labels
            )
            
            X_train_scaled = self.hog_scaler.fit_transform(X_train)
            X_test_scaled = self.hog_scaler.transform(X_test)
        else:
            X_train_scaled = self.hog_scaler.fit_transform(combined_features)
            X_test_scaled = X_train_scaled
            y_train = encoded_labels
            y_test = encoded_labels
        
        # Train SVM với optimization
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        self.hog_svm_model = grid_search.best_estimator_
        
        # Store ORB reference features
        self.orb_reference_features = orb_reference_features
        
        # Evaluate
        y_pred = self.hog_svm_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Ensemble training accuracy: {accuracy:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.hog_label_encoder.classes_))
        
        # Save model
        self._save_model()
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        result = {
            'accuracy': accuracy,
            'training_time': training_time,
            'num_samples': len(hog_features),
            'num_classes': len(self.hog_label_encoder.classes_),
            'classes': list(self.hog_label_encoder.classes_),
            'best_params': grid_search.best_params_,
            'feature_shape': combined_features.shape
        }
        
        print(f"✅ Ensemble training completed in {training_time:.2f}s")
        return result
    
    def predict(self, image: np.ndarray) -> EnsembleResult:
        """Ensemble prediction với HOG + ORB"""
        start_time = time.time()
        
        if not self.is_trained:
            return EnsembleResult(
                predicted_screen="ERROR", confidence=0.0, processing_time=time.time() - start_time,
                hog_prediction="ERROR", hog_confidence=0.0, orb_prediction="ERROR", orb_confidence=0.0,
                ensemble_weights={}
            )
        
        try:
            # Extract features
            features = self._extract_combined_features(image)
            
            # HOG + SVM prediction
            hog_prediction, hog_confidence = self._predict_hog(features)
            
            # ORB prediction
            orb_prediction, orb_confidence = self._predict_orb(features)
            
            # Ensemble decision với adaptive weights
            final_prediction, final_confidence, weights = self._ensemble_decision(
                hog_prediction, hog_confidence, orb_prediction, orb_confidence
            )
            
            processing_time = time.time() - start_time
            
            return EnsembleResult(
                predicted_screen=final_prediction,
                confidence=final_confidence,
                processing_time=processing_time,
                hog_prediction=hog_prediction,
                hog_confidence=hog_confidence,
                orb_prediction=orb_prediction,
                orb_confidence=orb_confidence,
                ensemble_weights=weights
            )
            
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            return EnsembleResult(
                predicted_screen="ERROR", confidence=0.0, processing_time=time.time() - start_time,
                hog_prediction="ERROR", hog_confidence=0.0, orb_prediction="ERROR", orb_confidence=0.0,
                ensemble_weights={}
            )
    
    def _predict_hog(self, features: Dict) -> Tuple[str, float]:
        """HOG + SVM prediction"""
        try:
            if features['hog'] is None:
                return "ERROR", 0.0
            
            # Combine features
            combined_features = np.hstack([
                features['hog'], 
                features['color_hist'], 
                [features['edge_density']]
            ])
            
            features_scaled = self.hog_scaler.transform([combined_features])
            
            prediction = self.hog_svm_model.predict(features_scaled)[0]
            probabilities = self.hog_svm_model.predict_proba(features_scaled)[0]
            
            predicted_screen = self.hog_label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            return predicted_screen, confidence
            
        except Exception as e:
            print(f"Error in HOG prediction: {e}")
            return "ERROR", 0.0
    
    def _predict_orb(self, features: Dict) -> Tuple[str, float]:
        """ORB-based prediction using feature matching"""
        try:
            if features['orb_descriptors'] is None:
                return "ERROR", 0.0
            
            best_match_screen = None
            best_match_score = 0.0
            
            # Match against all reference ORB features
            for screen_type, reference_descriptors_list in self.orb_reference_features.items():
                max_score_for_screen = 0.0
                
                for ref_descriptors in reference_descriptors_list:
                    score = self._match_orb_features(features['orb_descriptors'], ref_descriptors)
                    max_score_for_screen = max(max_score_for_screen, score)
                
                if max_score_for_screen > best_match_score:
                    best_match_score = max_score_for_screen
                    best_match_screen = screen_type
            
            if best_match_screen is None:
                return "ERROR", 0.0
            
            return best_match_screen, best_match_score
            
        except Exception as e:
            print(f"Error in ORB prediction: {e}")
            return "ERROR", 0.0
    
    def _ensemble_decision(self, hog_pred: str, hog_conf: float, orb_pred: str, orb_conf: float) -> Tuple[str, float, Dict]:
        """Adaptive ensemble decision making với Production boost v2.0"""
        
        # Special handling for Production screens
        production_boost = False
        if hog_pred == "Production" or orb_pred == "Production":
            production_boost = True
        
        # Adaptive weights based on confidence and screen type
        if production_boost and orb_pred == "Production":
            # If ORB predicts Production, trust it more (Production screens hard for HOG)
            weights = {'hog': 0.3, 'orb': 0.7}
            print(f"🎯 Production boost: ORB={orb_pred}({orb_conf:.3f}) vs HOG={hog_pred}({hog_conf:.3f})")
        elif production_boost and hog_pred == "Production":
            # If HOG predicts Production, keep normal high weight
            weights = {'hog': 0.85, 'orb': 0.15}
        elif hog_conf > 0.7:  # High HOG confidence
            weights = {'hog': 0.8, 'orb': 0.2}
        elif orb_conf > 0.7:  # High ORB confidence  
            weights = {'hog': 0.4, 'orb': 0.6}
        else:  # Balanced weights
            weights = {'hog': 0.7, 'orb': 0.3}  # Slightly favor HOG
        
        # Agreement check
        if hog_pred == orb_pred:
            # Both agree - boost confidence
            final_confidence = weights['hog'] * hog_conf + weights['orb'] * orb_conf
            final_confidence = min(1.0, final_confidence * 1.2)  # Higher boost for agreement
            return hog_pred, final_confidence, weights
        else:
            # Disagreement - use weighted decision
            hog_weighted_score = weights['hog'] * hog_conf
            orb_weighted_score = weights['orb'] * orb_conf
            
            # Additional Production bias for edge cases
            if production_boost:
                if hog_pred == "Production" and hog_conf > 0.1:
                    hog_weighted_score *= 1.2  # Moderate Production bias
                if orb_pred == "Production" and orb_conf > 0.1:
                    orb_weighted_score *= 1.5  # Stronger Production bias for ORB
            
            if hog_weighted_score > orb_weighted_score:
                return hog_pred, hog_weighted_score, weights
            else:
                return orb_pred, orb_weighted_score, weights
    
    def _save_model(self):
        """Save ensemble model"""
        try:
            model_data = {
                'hog_svm_model': self.hog_svm_model,
                'hog_scaler': self.hog_scaler,
                'hog_label_encoder': self.hog_label_encoder,
                'orb_reference_features': self.orb_reference_features,
                'ensemble_weights': self.ensemble_weights,
                'confidence_threshold': self.confidence_threshold,
                'hog_params': self.hog_params,
                'image_size': self.image_size,
                'is_trained': True
            }
            
            joblib.dump(model_data, self.model_save_path)
            print(f"💾 Ensemble model saved to {self.model_save_path}")
            
        except Exception as e:
            print(f"Error saving ensemble model: {e}")
    
    def _load_model(self):
        """Load ensemble model"""
        try:
            if os.path.exists(self.model_save_path):
                model_data = joblib.load(self.model_save_path)
                
                self.hog_svm_model = model_data['hog_svm_model']
                self.hog_scaler = model_data['hog_scaler']
                self.hog_label_encoder = model_data['hog_label_encoder']
                self.orb_reference_features = model_data['orb_reference_features']
                self.ensemble_weights = model_data['ensemble_weights']
                self.confidence_threshold = model_data['confidence_threshold']
                self.hog_params = model_data['hog_params']
                self.image_size = model_data['image_size']
                self.is_trained = model_data['is_trained']
                
                print(f"✅ Ensemble model loaded from {self.model_save_path}")
                print(f"📋 Classes: {list(self.hog_label_encoder.classes_)}")
                
        except Exception as e:
            print(f"Error loading ensemble model: {e}")
            self.is_trained = False

def main():
    """Demo ensemble classifier"""
    print("🚀 Ensemble HOG + ORB Classifier Demo")
    
    classifier = EnsembleHOGORBClassifier()
    
    training_folders = [r"D:\Wrembly\python_api_test\augmented_training_data"]
    
    if not classifier.is_trained:
        print("🔄 Training ensemble model...")
        result = classifier.train_from_folders(training_folders)
        print(f"Training result: {result}")
    else:
        print("✅ Using existing ensemble model")

if __name__ == "__main__":
    main() 