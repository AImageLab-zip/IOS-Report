#!/usr/bin/env python3
"""
Web-based Interactive Photo Labeling Tool

This tool helps label intraoral photos by displaying all 5 photos for each patient
in a web browser and allowing the user to click them in order: left, center, right, up, down.
The files will be automatically renamed based on the click order.

Multi-user support: Each user gets random patients to avoid conflicts.
"""

import os
import sys
import base64
import random
import threading
from pathlib import Path
from typing import List, Tuple, Optional
from flask import Flask, render_template_string, jsonify, request, session
from PIL import Image
import io


class PhotoLabelerWeb:
    """Web-based photo labeling application with multi-user support."""
    
    LABEL_ORDER = ['left', 'center', 'right', 'up', 'down']
    DISPLAY_SIZE = (400, 200)  # Width x Height for each thumbnail (reduced for faster loading)
    JPEG_QUALITY = 85  # JPEG quality for faster encoding
    
    def __init__(self, dataset_root: str):
        """
        Initialize the photo labeler.
        
        Args:
            dataset_root: Root directory of the dataset
        """
        self.dataset_root = Path(dataset_root)
        self.patient_folders = self._scan_patients()
        
        # Shuffle for multi-user support
        random.shuffle(self.patient_folders)
        
        # Thread-safe lock for file operations
        self.lock = threading.Lock()
        
        # Session-based state (stored per user)
        # Format: {session_id: {'patient_folder': Path, 'selected_indices': []}}
        self.user_sessions = {}
        
        # Track which patients are currently assigned to users (to avoid duplicates)
        self.assigned_patients = set()
        
        print(f"Found {len(self.patient_folders)} patients to label")
    
    def _scan_patients(self) -> List[Path]:
        """
        Scan the dataset for patient folders that need labeling.
        
        Returns:
            List of patient folder paths
        """
        patient_folders = []
        
        for patient_folder in sorted(self.dataset_root.iterdir()):
            if not patient_folder.is_dir():
                continue
            
            photos_dir = patient_folder / "intraoral-photos"
            if not photos_dir.exists():
                continue
            
            # Check if has unlabeled photos (a, b, c, d, e)
            unlabeled_photos = (list(photos_dir.glob("a.*")) + 
                               list(photos_dir.glob("b.*")) + 
                               list(photos_dir.glob("c.*")) + 
                               list(photos_dir.glob("d.*")) + 
                               list(photos_dir.glob("e.*")))
            
            if len(unlabeled_photos) == 5:
                patient_folders.append(patient_folder)
        
        return patient_folders
    
    def _is_patient_labeled(self, patient_folder: Path) -> bool:
        """
        Check if a patient is already labeled.
        
        Args:
            patient_folder: Path to patient folder
            
        Returns:
            True if already labeled
        """
        photos_dir = patient_folder / "intraoral-photos"
        labeled_photos = list(photos_dir.glob("left.*")) + list(photos_dir.glob("center.*"))
        return len(labeled_photos) > 0
    
    def _get_next_unlabeled_patient(self, exclude_assigned: bool = True) -> Optional[Path]:
        """
        Get the next unlabeled patient folder.
        Thread-safe: checks at the time of assignment to avoid conflicts.
        
        Args:
            exclude_assigned: If True, skip patients already assigned to other users
            
        Returns:
            Path to next unlabeled patient folder, or None if all done
        """
        with self.lock:
            # Re-check which patients still need labeling
            for patient_folder in self.patient_folders:
                # Skip if already labeled
                if self._is_patient_labeled(patient_folder):
                    continue
                
                # Skip if assigned to another user (unless override)
                if exclude_assigned and patient_folder in self.assigned_patients:
                    continue
                
                # Assign this patient
                self.assigned_patients.add(patient_folder)
                return patient_folder
            
            return None
    
    def _release_patient(self, patient_folder: Path):
        """
        Release a patient from the assigned set (e.g., after labeling complete).
        
        Args:
            patient_folder: Path to patient folder to release
        """
        with self.lock:
            self.assigned_patients.discard(patient_folder)
    
    def get_current_patient_data(self, session_id: str):
        """
        Get data for the current patient for a specific user session.
        
        Args:
            session_id: User session identifier
            
        Returns:
            Dictionary with patient data or None if all done
        """
        # Initialize session if needed
        if session_id not in self.user_sessions:
            patient_folder = self._get_next_unlabeled_patient(exclude_assigned=True)
            if patient_folder is None:
                return None
            self.user_sessions[session_id] = {
                'patient_folder': patient_folder,
                'selected_indices': []
            }
        
        session_data = self.user_sessions[session_id]
        patient_folder = session_data['patient_folder']
        
        # Double-check if patient is still unlabeled (might have been done by another user)
        if self._is_patient_labeled(patient_folder):
            # Release this patient and get next unlabeled patient
            self._release_patient(patient_folder)
            patient_folder = self._get_next_unlabeled_patient(exclude_assigned=True)
            if patient_folder is None:
                # Clean up session
                del self.user_sessions[session_id]
                return None
            session_data['patient_folder'] = patient_folder
            session_data['selected_indices'] = []
        
        photos_dir = patient_folder / "intraoral-photos"
        
        # Load images
        images_data = []
        for letter in ['a', 'b', 'c', 'd', 'e']:
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                img_path = photos_dir / f"{letter}.{ext}"
                if img_path.exists():
                    # Load and resize image
                    img = Image.open(img_path)
                    # Convert to RGB if needed (removes alpha channel)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.thumbnail(self.DISPLAY_SIZE, Image.Resampling.LANCZOS)
                    
                    # Convert to base64 with compression
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=self.JPEG_QUALITY, optimize=True)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    images_data.append({
                        'letter': letter,
                        'path': str(img_path),
                        'data': f"data:image/jpeg;base64,{img_base64}"
                    })
                    break
        
        # Count remaining patients (unlabeled and not assigned)
        with self.lock:
            remaining = sum(1 for pf in self.patient_folders 
                          if not self._is_patient_labeled(pf) and pf not in self.assigned_patients)
        
        return {
            'patient_name': patient_folder.name,
            'remaining_patients': remaining,
            'images': images_data,
            'selected': session_data['selected_indices']
        }
    
    def select_image(self, session_id: str, image_idx: int):
        """
        Select an image at the given index for a user session.
        This only updates the selection state, doesn't rename files.
        
        Args:
            session_id: User session identifier
            image_idx: Index of the image (0-4)
            
        Returns:
            Dictionary with status and message
        """
        if session_id not in self.user_sessions:
            return {'status': 'error', 'message': 'Session expired. Please refresh.'}
        
        session_data = self.user_sessions[session_id]
        selected_indices = session_data['selected_indices']
        
        if image_idx in selected_indices:
            return {'status': 'error', 'message': 'Image already selected'}
        
        selected_indices.append(image_idx)
        label = self.LABEL_ORDER[len(selected_indices) - 1]
        
        return {
            'status': 'ok',
            'message': f'Selected as "{label}" ({len(selected_indices)}/5)',
            'selected_count': len(selected_indices),
            'all_selected': len(selected_indices) == 5
        }
    
    def complete_patient(self, session_id: str):
        """
        Complete the current patient by renaming files and moving to next.
        Called after all 5 images are selected.
        
        Args:
            session_id: User session identifier
            
        Returns:
            Dictionary with status and message
        """
        if session_id not in self.user_sessions:
            return {'status': 'error', 'message': 'Session expired. Please refresh.'}
        
        session_data = self.user_sessions[session_id]
        selected_indices = session_data['selected_indices']
        patient_folder = session_data['patient_folder']
        
        # Verify all 5 images are selected
        if len(selected_indices) != 5:
            return {'status': 'error', 'message': f'Only {len(selected_indices)}/5 images selected'}
        
        # Rename files
        success = self._rename_files(patient_folder, selected_indices)
        
        if not success:
            # Release the patient and clear session
            self._release_patient(patient_folder)
            del self.user_sessions[session_id]
            return {'status': 'error', 'message': 'Failed to rename files. Another user may have already labeled this patient.'}
        
        # Release this patient from assigned set
        self._release_patient(patient_folder)
        
        # Get next patient
        next_patient = self._get_next_unlabeled_patient(exclude_assigned=True)
        
        if next_patient is None:
            # Clean up session
            del self.user_sessions[session_id]
            return {'status': 'complete', 'message': 'All patients labeled!'}
        else:
            # Update session with next patient
            session_data['patient_folder'] = next_patient
            session_data['selected_indices'] = []
            return {'status': 'next', 'message': 'Patient complete! Loading next patient...'}
    
    def reset_current(self, session_id: str):
        """
        Reset the current patient selection for a user session.
        Fast operation - only clears the selection array.
        
        Args:
            session_id: User session identifier
            
        Returns:
            Dictionary with status
        """
        if session_id in self.user_sessions:
            self.user_sessions[session_id]['selected_indices'] = []
            return {'status': 'ok', 'message': 'Selection reset'}
        return {'status': 'error', 'message': 'Session not found'}
    
    def _rename_files(self, patient_folder: Path, selected_indices: List[int]) -> bool:
        """
        Rename the files based on the selected order.
        Thread-safe with lock.
        
        Args:
            patient_folder: Path to patient folder
            selected_indices: List of selected image indices in order
            
        Returns:
            True if successful, False if files were already renamed
        """
        with self.lock:
            # Double-check patient is still unlabeled
            if self._is_patient_labeled(patient_folder):
                return False
            
            photos_dir = patient_folder / "intraoral-photos"
            
            # Get current image paths
            image_paths = []
            for letter in ['a', 'b', 'c', 'd', 'e']:
                for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                    img_path = photos_dir / f"{letter}.{ext}"
                    if img_path.exists():
                        image_paths.append(img_path)
                        break
            
            if len(image_paths) != 5:
                return False
            
            # Rename based on selection order
            for order_idx, img_idx in enumerate(selected_indices):
                old_path = image_paths[img_idx]
                label = self.LABEL_ORDER[order_idx]
                ext = old_path.suffix
                new_path = old_path.parent / f"{label}{ext}"
                
                old_path.rename(new_path)
                print(f"  Renamed: {old_path.name} -> {new_path.name}")
            
            print(f"✓ Completed patient: {patient_folder.name}")
            return True
    
    def get_labeled_images_by_type(self, image_type: str) -> List[dict]:
        """
        Get all labeled images of a specific type (left, right, center, up, down).
        
        Args:
            image_type: Type of image to retrieve (left, right, center, up, down)
            
        Returns:
            List of dictionaries with patient info and base64 image data
        """
        if image_type not in self.LABEL_ORDER:
            return []
        
        labeled_images = []
        
        for patient_folder in sorted(self.dataset_root.iterdir()):
            if not patient_folder.is_dir():
                continue
            
            photos_dir = patient_folder / "intraoral-photos"
            if not photos_dir.exists():
                continue
            
            # Look for the specific image type
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                img_path = photos_dir / f"{image_type}.{ext}"
                if img_path.exists():
                    # Load and encode image
                    try:
                        img = Image.open(img_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Create thumbnail for gallery view
                        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                        
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=85, optimize=True)
                        img_base64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        labeled_images.append({
                            'patient_id': patient_folder.name,
                            'image_data': f"data:image/jpeg;base64,{img_base64}"
                        })
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                    
                    break
        
        return labeled_images


# HTML template for gallery view
GALLERY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{title}} - Image Gallery</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            max-width: 1400px;
            margin: 0 auto 20px;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin: 0 0 10px 0;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 18px;
        }
        .nav {
            text-align: center;
            margin: 20px 0;
        }
        .nav a {
            display: inline-block;
            padding: 10px 20px;
            margin: 0 5px;
            background-color: #2196F3;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .nav a:hover {
            background-color: #1976D2;
        }
        .nav a.active {
            background-color: #4CAF50;
        }
        .nav a.home {
            background-color: #ff9800;
        }
        .nav a.home:hover {
            background-color: #f57c00;
        }
        .gallery {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            padding: 20px;
        }
        .gallery-item {
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .gallery-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .gallery-item img {
            width: 100%;
            height: auto;
            display: block;
        }
        .gallery-item .caption {
            padding: 10px;
            text-align: center;
            font-weight: bold;
            color: #333;
            background-color: #f9f9f9;
        }
        .empty {
            text-align: center;
            padding: 40px;
            color: #999;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{title}}</h1>
        <div class="subtitle">{{count}} images labeled</div>
        <div class="nav">
            <a href="/" class="home">🏠 Home (Labeling Tool)</a>
            <a href="/left" {% if view_type == 'left' %}class="active"{% endif %}>👈 Left</a>
            <a href="/center" {% if view_type == 'center' %}class="active"{% endif %}>🎯 Center</a>
            <a href="/right" {% if view_type == 'right' %}class="active"{% endif %}>👉 Right</a>
            <a href="/up" {% if view_type == 'up' %}class="active"{% endif %}>👆 Up</a>
            <a href="/down" {% if view_type == 'down' %}class="active"{% endif %}>👇 Down</a>
        </div>
    </div>
    
    {% if images %}
    <div class="gallery">
        {% for img in images %}
        <div class="gallery-item">
            <img src="{{img.image_data}}" alt="{{img.patient_id}}">
            <div class="caption">{{img.patient_id}}</div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <div class="empty">
        No {{view_type}} images have been labeled yet.
    </div>
    {% endif %}
</body>
</html>
"""

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Photo Labeling Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .progress {
            text-align: center;
            font-size: 18px;
            margin-bottom: 20px;
            color: #666;
        }
        .instruction {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #e3f2fd;
            border-radius: 5px;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        .image-container {
            position: relative;
            border: 3px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.3s;
        }
        .image-container:hover {
            border-color: #2196F3;
            transform: scale(1.02);
        }
        .image-container.selected {
            border-color: #4CAF50;
            border-width: 5px;
        }
        .image-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        .image-label {
            position: absolute;
            top: 10px;
            left: 10px;
            background-color: rgba(255, 0, 0, 0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 18px;
        }
        .selection-label {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background-color: rgba(76, 175, 80, 0.9);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 16px;
        }
        .controls {
            text-align: center;
            margin-top: 20px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            margin: 0 5px;
        }
        .reset-btn {
            background-color: #ff9800;
            color: white;
        }
        .reset-btn:hover {
            background-color: #f57c00;
        }
        .message {
            text-align: center;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
            display: none;
        }
        .message.success {
            background-color: #c8e6c9;
            color: #2e7d32;
        }
        .message.error {
            background-color: #ffcdd2;
            color: #c62828;
        }
        .complete {
            text-align: center;
            font-size: 24px;
            color: #4CAF50;
            padding: 40px;
        }
        /* Spacing for grid */
        .image-container:nth-child(4) {
            grid-column: 1;
        }
        .image-container:nth-child(5) {
            grid-column: 3;
        }
        .stats {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.9);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .loading-overlay.active {
            display: flex;
        }
        .loading-content {
            text-align: center;
        }
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #2196F3;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-text {
            font-size: 18px;
            color: #333;
            font-weight: bold;
        }
        .completion-animation {
            animation: fadeInScale 0.5s ease-in-out;
        }
        @keyframes fadeInScale {
            0% {
                opacity: 0;
                transform: scale(0.8);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 Intraoral Photo Labeling Tool</h1>
        
        <!-- Navigation to galleries -->
        <div style="text-align: center; margin-bottom: 20px; padding: 10px; background-color: #e8f5e9; border-radius: 5px;">
            <strong>View Labeled Images:</strong>
            <a href="/left" style="margin: 0 8px; color: #1976D2; text-decoration: none; font-weight: bold;">👈 Left</a>
            <a href="/center" style="margin: 0 8px; color: #1976D2; text-decoration: none; font-weight: bold;">🎯 Center</a>
            <a href="/right" style="margin: 0 8px; color: #1976D2; text-decoration: none; font-weight: bold;">👉 Right</a>
            <a href="/up" style="margin: 0 8px; color: #1976D2; text-decoration: none; font-weight: bold;">👆 Up</a>
            <a href="/down" style="margin: 0 8px; color: #1976D2; text-decoration: none; font-weight: bold;">👇 Down</a>
        </div>
        
        <div id="content">
            <div class="progress" id="progress"></div>
            <div class="instruction" id="instruction"></div>
            
            <div class="image-grid" id="imageGrid"></div>
            
            <div class="controls">
                <button class="reset-btn" onclick="resetSelection()">🔄 Reset Selection</button>
            </div>
            
            <div class="stats" id="stats"></div>
            
            <div class="message" id="message"></div>
        </div>
    </div>
    
    <!-- Loading overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="spinner"></div>
            <div class="loading-text">Loading next patient...</div>
        </div>
    </div>

    <script>
        let currentData = null;

        function showLoading() {
            document.getElementById('loadingOverlay').classList.add('active');
        }

        function hideLoading() {
            document.getElementById('loadingOverlay').classList.remove('active');
        }

        function loadPatientData() {
            showLoading();
            fetch('/get_patient')
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    if (data === null) {
                        document.getElementById('content').innerHTML = 
                            '<div class="complete">✅ All patients have been labeled!</div>';
                        return;
                    }
                    
                    currentData = data;
                    updateDisplay(true); // Animate when loading new patient
                })
                .catch(error => {
                    hideLoading();
                    showMessage('Error loading patient data: ' + error, 'error');
                });
        }

        function updateDisplay(animate = false) {
            if (currentData === null) return;

            // Update progress
            document.getElementById('progress').textContent = 
                `Patient: ${currentData.patient_name}`;
            
            // Update stats
            document.getElementById('stats').textContent = 
                `Remaining patients: ${currentData.remaining_patients}`;

            // Update instruction
            const numSelected = currentData.selected.length;
            const labelOrder = ['left', 'center', 'right', 'up', 'down'];
            
            const instructionEl = document.getElementById('instruction');
            if (numSelected < 5) {
                instructionEl.innerHTML = `Click the "${labelOrder[numSelected]}" image (${numSelected + 1}/5)`;
            } else {
                // Show complete button when all selected
                instructionEl.innerHTML = `
                    <span style="color: #4CAF50;">✓ All 5 images selected!</span>
                    <button onclick="completePatient()" 
                            style="margin-left: 20px; padding: 10px 30px; background-color: #4CAF50; 
                                   color: white; border: none; border-radius: 5px; cursor: pointer; 
                                   font-size: 16px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                        ✓ Complete & Next Patient
                    </button>
                `;
            }

            // Update image grid (only animate when loading new patient)
            const grid = document.getElementById('imageGrid');
            grid.innerHTML = '';
            if (animate) {
                grid.classList.add('completion-animation');
            }

            currentData.images.forEach((img, idx) => {
                const container = document.createElement('div');
                container.className = 'image-container';
                if (currentData.selected.includes(idx)) {
                    container.className += ' selected';
                }
                // Only allow clicking if not all selected yet
                if (numSelected < 5 && !currentData.selected.includes(idx)) {
                    container.onclick = () => selectImage(idx);
                } else if (numSelected < 5) {
                    container.style.cursor = 'not-allowed';
                } else {
                    container.style.cursor = 'default';
                }

                const imgEl = document.createElement('img');
                imgEl.src = img.data;
                imgEl.alt = img.letter;

                const label = document.createElement('div');
                label.className = 'image-label';
                label.textContent = img.letter.toUpperCase();

                container.appendChild(imgEl);
                container.appendChild(label);

                // Add selection label if selected
                if (currentData.selected.includes(idx)) {
                    const selectionIdx = currentData.selected.indexOf(idx);
                    const selectionLabel = document.createElement('div');
                    selectionLabel.className = 'selection-label';
                    selectionLabel.textContent = labelOrder[selectionIdx].toUpperCase();
                    container.appendChild(selectionLabel);
                }

                grid.appendChild(container);
            });
            
            // Remove animation class after animation completes (only if animated)
            if (animate) {
                setTimeout(() => {
                    grid.classList.remove('completion-animation');
                }, 500);
            }
        }

        function selectImage(idx) {
            // Quick local update first for instant feedback
            if (currentData.selected.includes(idx)) {
                showMessage('Image already selected', 'error');
                return;
            }
            
            // Update UI immediately
            currentData.selected.push(idx);
            updateDisplay();
            
            // Send to server (no loading needed)
            fetch('/select_image', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: idx})
            })
            .then(response => response.json())
            .then(result => {
                if (result.status === 'error') {
                    // Revert local change
                    currentData.selected.pop();
                    updateDisplay();
                    showMessage(result.message, 'error');
                } else if (result.status === 'ok') {
                    showMessage(result.message, 'success');
                    
                    // If all selected, show complete button
                    if (result.all_selected) {
                        showCompleteButton();
                    }
                }
            })
            .catch(error => {
                // Revert local change
                currentData.selected.pop();
                updateDisplay();
                showMessage('Error: ' + error, 'error');
            });
        }
        
        function showCompleteButton() {
            const instruction = document.getElementById('instruction');
            instruction.innerHTML = `
                <span style="color: #4CAF50;">✓ All 5 images selected!</span>
                <button onclick="completePatient()" 
                        style="margin-left: 20px; padding: 10px 30px; background-color: #4CAF50; 
                               color: white; border: none; border-radius: 5px; cursor: pointer; 
                               font-size: 16px; font-weight: bold;">
                    ✓ Complete & Next Patient
                </button>
            `;
        }
        
        function completePatient() {
            showLoading();
            
            fetch('/complete_patient', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(result => {
                if (result.status === 'error') {
                    hideLoading();
                    showMessage(result.message, 'error');
                    if (result.message.includes('already labeled')) {
                        setTimeout(() => {
                            showLoading();
                            loadPatientData();
                        }, 2000);
                    }
                } else if (result.status === 'next' || result.status === 'complete') {
                    showMessage(result.message, 'success');
                    setTimeout(() => {
                        loadPatientData();
                    }, 500);
                }
            })
            .catch(error => {
                hideLoading();
                showMessage('Error: ' + error, 'error');
            });
        }

        function resetSelection() {
            fetch('/reset', {method: 'POST'})
                .then(response => response.json())
                .then(result => {
                    showMessage(result.message, 'success');
                    // Just reload the current data (fast, no file operations)
                    loadPatientData();
                })
                .catch(error => {
                    showMessage('Error: ' + error, 'error');
                });
        }

        function showMessage(text, type) {
            const msg = document.getElementById('message');
            msg.textContent = text;
            msg.className = 'message ' + type;
            msg.style.display = 'block';
            setTimeout(() => {
                msg.style.display = 'none';
            }, 3000);
        }

        // Load initial data
        loadPatientData();
    </script>
</body>
</html>
"""


def create_app(dataset_root: str):
    """Create and configure the Flask app."""
    app = Flask(__name__)
    app.secret_key = os.urandom(24)  # For session management
    labeler = PhotoLabelerWeb(dataset_root)
    
    @app.route('/')
    def index():
        # Generate session ID if not exists
        if 'user_id' not in session:
            session['user_id'] = os.urandom(16).hex()
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/get_patient')
    def get_patient():
        user_id = session.get('user_id', 'default')
        return jsonify(labeler.get_current_patient_data(user_id))
    
    @app.route('/select_image', methods=['POST'])
    def select_image():
        user_id = session.get('user_id', 'default')
        data = request.json
        result = labeler.select_image(user_id, data['index'])
        return jsonify(result)
    
    @app.route('/complete_patient', methods=['POST'])
    def complete_patient():
        user_id = session.get('user_id', 'default')
        result = labeler.complete_patient(user_id)
        return jsonify(result)
    
    @app.route('/reset', methods=['POST'])
    def reset():
        user_id = session.get('user_id', 'default')
        result = labeler.reset_current(user_id)
        return jsonify(result)
    
    # Gallery view endpoints
    @app.route('/left')
    def view_left():
        images = labeler.get_labeled_images_by_type('left')
        return render_template_string(GALLERY_TEMPLATE, 
                                     title='Left View Images',
                                     view_type='left',
                                     images=images,
                                     count=len(images))
    
    @app.route('/right')
    def view_right():
        images = labeler.get_labeled_images_by_type('right')
        return render_template_string(GALLERY_TEMPLATE,
                                     title='Right View Images',
                                     view_type='right',
                                     images=images,
                                     count=len(images))
    
    @app.route('/center')
    def view_center():
        images = labeler.get_labeled_images_by_type('center')
        return render_template_string(GALLERY_TEMPLATE,
                                     title='Center View Images',
                                     view_type='center',
                                     images=images,
                                     count=len(images))
    
    @app.route('/up')
    def view_up():
        images = labeler.get_labeled_images_by_type('up')
        return render_template_string(GALLERY_TEMPLATE,
                                     title='Up View Images',
                                     view_type='up',
                                     images=images,
                                     count=len(images))
    
    @app.route('/down')
    def view_down():
        images = labeler.get_labeled_images_by_type('down')
        return render_template_string(GALLERY_TEMPLATE,
                                     title='Down View Images',
                                     view_type='down',
                                     images=images,
                                     count=len(images))
    
    return app


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
    else:
        # Default to the dataset location
        dataset_root = "/work/grana_maxillo/Dataset_FerraraDump_ToothFairy4M"
    
    print(f"Starting Web-based Photo Labeling Tool")
    print(f"Dataset: {dataset_root}\n")
    
    app = create_app(dataset_root)
    
    print("\n" + "="*60)
    print("🌐 Server starting...")
    print("="*60)
    print("\n✨ Features:")
    print("  • Multi-user support (each user gets different patients)")
    print("  • Automatic conflict prevention")
    print("  • Optimized image loading for speed")
    print("\n📍 Access the labeling tool in your browser at:")
    print("  http://localhost:5000")
    print("\nOr from your Windows machine:")
    print("  http://<cluster-hostname>:5000")
    print("\n💡 Instructions:")
    print("  1. Open the URL in your browser")
    print("  2. Click images in order: left, center, right, up, down")
    print("  3. Files auto-rename and next patient loads automatically")
    print("  4. Use 'Reset Selection' button if you make a mistake")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
