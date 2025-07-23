#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 HOG + SVM Classifier cho HMI Screen Detection
Sử dụng Histogram of Oriented Gradients với Support Vector Machine
để phân loại các loại màn hình HMI khác nhau

Features:
- HOG feature extraction được tối ưu hóa
- SVM classifier với kernel RBF
- Training pipeline tự động
- Fast prediction với caching
- Robust preprocessing

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ClassificationResult:
    """Kết quả phân loại"""
    predicted_screen: str
    confidence: float
    processing_time: float
    hog_features: Optional[np.ndarray] = None

class HOGSVMClassifier:
    """
    HOG + SVM Classifier cho HMI Screen Detection
    Tối ưu hóa cho tốc độ và độ chính xác cao
    """
    
    def __init__(self, model_save_path: str = "hog_svm_model.pkl"):
        self.model_save_path = model_save_path
        self.svm_model = None
        self.scaler = None
        self.label_encoder = None
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'transform_sqrt': True,
            'feature_vector': True
        }
        self.image_size = (128, 128)  # Standard size for HOG
        self.is_trained = False
        
        # Load model if exists
        self._load_model()
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh cho HOG extraction
        """
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
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(resized)
            
            # Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
            
            return blurred / 255.0  # Normalize back to [0,1]
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
    
    def _extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features từ ảnh đã tiền xử lý
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            if processed_image is None:
                return None
            
            # Extract HOG features
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
    
    def _extract_screen_from_filename(self, filename: str) -> str:
        """
        Extract screen type từ filename - F1 NEW FORMAT ONLY
        """
        try:
            # NEW FORMAT: IE-F1-CWA01_Production_Data.jpg
            if '_' in filename and not filename.startswith('template_'):
                base_name = filename.replace('.jpg', '').replace('.png', '')
                parts = base_name.split('_', 1)
                if len(parts) >= 2:
                    screen_type = parts[1]
                    return screen_type
            
            return 'Unknown'
                
        except Exception as e:
            print(f"Error extracting screen from filename {filename}: {e}")
            return 'Unknown'
    
    def train_from_folders(self, folder_paths: List[str], test_size: float = 0.1) -> Dict:
        """
        Train model từ các folder chứa ảnh
        """
        print("🚀 Starting HOG + SVM Training với Reference Images")
        start_time = time.time()
        
        # Collect all images and labels
        images = []
        labels = []
        
        for folder_path in folder_paths:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                print(f"⚠️ Folder không tồn tại: {folder_path}")
                continue
            
            print(f"📁 Processing folder: {folder_path}")
            
            # Find all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
                image_files.extend(folder_path.glob(f"*{ext.upper()}"))
            
            print(f"🔍 Found {len(image_files)} images")
            
            for image_file in image_files:
                try:
                    # Load image
                    image = cv2.imread(str(image_file))
                    if image is None:
                        continue
                    
                    # Extract screen type from filename
                    screen_type = self._extract_screen_from_filename(image_file.name)
                    if screen_type == 'Unknown':
                        print(f"⚠️ Skipping unknown screen type: {image_file.name}")
                        continue
                    
                    images.append(image)
                    labels.append(screen_type)
                    print(f"✅ Added {image_file.name} -> {screen_type}")
                    
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    continue
        
        if len(images) == 0:
            print("❌ Không có ảnh hợp lệ để training")
            return {'error': 'No valid images for training'}
        
        print(f"📊 Total images for training: {len(images)}")
        unique_labels = list(set(labels))
        print(f"📋 Screen types ({len(unique_labels)}): {unique_labels}")
        
        # Check if we have enough data for train_test_split
        if len(unique_labels) > len(images) * (1 - test_size):
            print(f"⚠️ Too many classes for test_size, adjusting to single-fold training")
            use_split = False
        else:
            use_split = True
        
        # Extract HOG features
        print("🔄 Extracting HOG features...")
        features = []
        valid_labels = []
        
        for i, (image, label) in enumerate(zip(images, labels)):
            if i % 5 == 0:
                print(f"Progress: {i+1}/{len(images)}")
            
            hog_features = self._extract_hog_features(image)
            if hog_features is not None:
                features.append(hog_features)
                valid_labels.append(label)
        
        if len(features) == 0:
            print("❌ Không extract được HOG features")
            return {'error': 'No valid HOG features'}
        
        features = np.array(features)
        print(f"✅ Extracted features shape: {features.shape}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(valid_labels)
        
        # Scale features
        print("🔄 Scaling features...")
        self.scaler = StandardScaler()
        
        if use_split:
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                features, encoded_labels, test_size=test_size, 
                random_state=42, stratify=encoded_labels
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            # Use all data for training (no test split)
            X_train_scaled = self.scaler.fit_transform(features)
            X_test_scaled = X_train_scaled
            y_train = encoded_labels
            y_test = encoded_labels
        
        # Train SVM with simpler Grid Search for small datasets
        print("🔄 Training SVM...")
        if len(features) < 50:  # Small dataset
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        else:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
        
        svm = SVC(probability=True, random_state=42)
        
        if len(features) < 20:  # Very small dataset
            # No grid search, use default parameters
            self.svm_model = svm
            self.svm_model.fit(X_train_scaled, y_train)
            best_params = {'default': 'used'}
        else:
            grid_search = GridSearchCV(
                svm, param_grid, cv=min(3, len(unique_labels)), scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.svm_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        
        print(f"✅ Best parameters: {best_params}")
        
        # Evaluate model
        y_pred = self.svm_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Test accuracy: {accuracy:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Save model
        self._save_model()
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        result = {
            'accuracy': accuracy,
            'training_time': training_time,
            'num_samples': len(features),
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'best_params': best_params,
            'feature_shape': features.shape,
            'use_split': use_split
        }
        
        print(f"✅ Training completed in {training_time:.2f}s")
        return result
    
    def predict(self, image: np.ndarray) -> ClassificationResult:
        """
        Predict screen type cho một ảnh
        """
        start_time = time.time()
        
        if not self.is_trained or self.svm_model is None:
            return ClassificationResult(
                predicted_screen="ERROR",
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        try:
            # Extract HOG features
            hog_features = self._extract_hog_features(image)
            if hog_features is None:
                return ClassificationResult(
                    predicted_screen="ERROR",
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Scale features
            features_scaled = self.scaler.transform([hog_features])
            
            # Predict
            prediction = self.svm_model.predict(features_scaled)[0]
            probabilities = self.svm_model.predict_proba(features_scaled)[0]
            
            # Get screen name and confidence
            predicted_screen = self.label_encoder.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            processing_time = time.time() - start_time
            
            return ClassificationResult(
                predicted_screen=predicted_screen,
                confidence=confidence,
                processing_time=processing_time,
                hog_features=hog_features
            )
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return ClassificationResult(
                predicted_screen="ERROR",
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def predict_from_file(self, image_path: str) -> ClassificationResult:
        """
        Predict từ file ảnh
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return ClassificationResult(
                    predicted_screen="ERROR",
                    confidence=0.0,
                    processing_time=0.0
                )
            
            return self.predict(image)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return ClassificationResult(
                predicted_screen="ERROR",
                confidence=0.0,
                processing_time=0.0
            )
    
    def _save_model(self):
        """
        Lưu model đã train
        """
        try:
            model_data = {
                'svm_model': self.svm_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'hog_params': self.hog_params,
                'image_size': self.image_size,
                'is_trained': True
            }
            
            joblib.dump(model_data, self.model_save_path)
            print(f"💾 Model saved to {self.model_save_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _load_model(self):
        """
        Load model đã train
        """
        try:
            if os.path.exists(self.model_save_path):
                model_data = joblib.load(self.model_save_path)
                
                self.svm_model = model_data['svm_model']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.hog_params = model_data['hog_params']
                self.image_size = model_data['image_size']
                self.is_trained = model_data['is_trained']
                
                print(f"✅ Model loaded from {self.model_save_path}")
                print(f"📋 Classes: {list(self.label_encoder.classes_)}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_trained = False
    
    def get_feature_importance(self) -> Dict:
        """
        Lấy thông tin về feature importance (cho SVM linear)
        """
        if not self.is_trained or self.svm_model is None:
            return {}
        
        try:
            if hasattr(self.svm_model, 'coef_'):
                feature_importance = np.abs(self.svm_model.coef_).mean(axis=0)
                return {
                    'feature_importance': feature_importance,
                    'top_features': np.argsort(feature_importance)[-10:][::-1]
                }
        except Exception as e:
            print(f"Error getting feature importance: {e}")
        
        return {}

def main():
    """
    Demo usage của HOG + SVM Classifier
    """
    print("🚀 HOG + SVM Classifier Demo")
    
    # Khởi tạo classifier
    classifier = HOGSVMClassifier()
    
    # Training folders
    training_folders = [
        r"D:\Wrembly\nut",
        r"D:\Wrembly\nut1"
    ]
    
    # Check if model exists
    if not classifier.is_trained:
        print("🔄 Training new model...")
        result = classifier.train_from_folders(training_folders)
        print(f"Training result: {result}")
    else:
        print("✅ Using existing trained model")
    
    # Test prediction trên một ảnh
    test_folders = [r"D:\Wrembly\nut1"]
    for folder in test_folders:
        folder_path = Path(folder)
        if folder_path.exists():
            image_files = list(folder_path.glob("*.jpg"))[:3]  # Test 3 ảnh đầu
            
            for image_file in image_files:
                print(f"\n📸 Testing: {image_file.name}")
                result = classifier.predict_from_file(str(image_file))
                print(f"   Predicted: {result.predicted_screen}")
                print(f"   Confidence: {result.confidence:.4f}")
                print(f"   Time: {result.processing_time:.3f}s")

if __name__ == "__main__":
    main() 