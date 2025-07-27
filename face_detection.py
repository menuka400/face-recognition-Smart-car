import cv2
import numpy as np
import os
import threading
import time
from ultralytics import YOLO
from queue import Queue
import pickle

class FaceDetector:
    def __init__(self, model_path, save_folder, config_manager=None):
        self.model = YOLO(model_path)
        self.save_folder = save_folder
        self.config_manager = config_manager
        
        # Load configuration values
        if config_manager:
            self.confidence_threshold = config_manager.get('face_detection.confidence_threshold', 0.6)
            self.unknown_folder = config_manager.get('face_detection.unknown_folder', 'unknown_faces')
            self.camera_device_id = config_manager.get('camera.device_id', 0)
            self.camera_width = config_manager.get('camera.resolution.width', 640)
            self.camera_height = config_manager.get('camera.resolution.height', 480)
            self.camera_fps = config_manager.get('camera.fps', 30)
            self.max_images_per_folder = config_manager.get('face_detection.max_images_per_folder', 10)
            
            # Display settings
            self.window_name = config_manager.get('display.window_name', 'Face Recognition')
            self.show_confidence = config_manager.get('display.show_confidence', True)
            self.show_bounding_box = config_manager.get('display.show_bounding_box', True)
            self.font_scale = config_manager.get('display.font_scale', 0.6)
            self.font_thickness = config_manager.get('display.font_thickness', 2)
            self.box_thickness = config_manager.get('display.box_thickness', 2)
            
            # Colors
            self.known_face_color = tuple(config_manager.get('colors.known_face_box', [0, 255, 0]))
            self.unknown_face_color = tuple(config_manager.get('colors.unknown_face_box', [0, 0, 255]))
            self.text_bg_color = tuple(config_manager.get('colors.text_background', [0, 0, 0]))
            self.text_color = tuple(config_manager.get('colors.text_color', [255, 255, 255]))
        else:
            # Default values
            self.confidence_threshold = 0.6
            self.unknown_folder = "unknown_faces"
            self.camera_device_id = 0
            self.camera_width = 640
            self.camera_height = 480
            self.camera_fps = 30
            self.max_images_per_folder = 10
            self.window_name = "Face Recognition"
            self.show_confidence = True
            self.show_bounding_box = True
            self.font_scale = 0.6
            self.font_thickness = 2
            self.box_thickness = 2
            self.known_face_color = (0, 255, 0)
            self.unknown_face_color = (0, 0, 255)
            self.text_bg_color = (0, 0, 0)
            self.text_color = (255, 255, 255)
        self.face_queue = Queue()
        self.recognition_results = Queue()
        self.cleanup_queue = Queue()
        self.face_arrays = []
        self.running = False
        self.face_counter = 0
        self.saved_faces = {}
        self.saved_unknown_faces = {}
        self.current_recognitions = {}
        
        # Create save folders if they don't exist
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(self.unknown_folder, exist_ok=True)
        
    def detect_faces(self):
        """Main face detection loop with smart cleanup (silent mode)"""
        self.running = True
        cap = cv2.VideoCapture(self.camera_device_id)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {self.camera_device_id}")
            return
        
        # Set camera resolution and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        
        print(f"📷 Camera initialized: {self.camera_width}x{self.camera_height} @ {self.camera_fps}fps")
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=self.cleanup_worker, daemon=True)
        cleanup_thread.start()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Check for new recognition results and cleanup requests
            self.update_recognition_results()
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Draw face boxes and names
            display_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        
                        # Only process if confidence is high enough
                        if conf > self.confidence_threshold:
                            # Calculate face dimensions
                            face_width = x2 - x1
                            face_height = y2 - y1
                            
                            # Add generous padding (50% of face size)
                            padding_x = int(face_width * 0.5)
                            padding_y = int(face_height * 0.5)
                            
                            # Apply padding with bounds checking
                            x1_padded = max(0, x1 - padding_x)
                            y1_padded = max(0, y1 - padding_y)
                            x2_padded = min(frame.shape[1], x2 + padding_x)
                            y2_padded = min(frame.shape[0], y2 + padding_y)
                            
                            # Extract face region with padding
                            face_img = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                            
                            # Check if face is large enough
                            if (face_img.size > 0 and 
                                face_img.shape[0] > 100 and 
                                face_img.shape[1] > 100):
                                
                                # Resize face to a standard size for better recognition
                                face_resized = cv2.resize(face_img, (224, 224))
                                
                                # Save original and resized face
                                face_id = self.save_face(face_img, face_resized)
                                
                                # Add resized face to queue for recognition
                                face_data = {
                                    'image': face_resized.copy(),
                                    'face_id': face_id,
                                    'bbox': (x1, y1, x2, y2),
                                    'timestamp': time.time()
                                }
                                self.face_queue.put(face_data)
                                
                                # Draw face rectangle with configured colors
                                person_name, confidence = self.get_recognition_for_area(x1, y1, x2, y2)
                                
                                if self.show_bounding_box:
                                    # Choose color based on recognition result
                                    if person_name and person_name != "UNKNOWN":
                                        color = self.known_face_color
                                    else:
                                        color = self.unknown_face_color
                                    
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, self.box_thickness)
                                
                                # Check if we have recognition result for this area
                                if person_name:
                                    # Display person name and confidence
                                    label = f"{person_name}"
                                    confidence_label = f"{confidence:.1f}%" if confidence > 0 and self.show_confidence else ""
                                    
                                    # Calculate text size for background
                                    (text_width, text_height), baseline = cv2.getTextSize(
                                        label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
                                    (conf_width, conf_height), _ = cv2.getTextSize(
                                        confidence_label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, self.font_thickness)
                                    
                                    # Draw background rectangles for text
                                    bg_color = self.known_face_color if person_name != "UNKNOWN" else self.unknown_face_color
                                    cv2.rectangle(display_frame, 
                                                (x1, y1 - text_height - 35), 
                                                (x1 + max(text_width, conf_width) + 10, y1), 
                                                bg_color, -1)
                                    
                                    # Draw person name
                                    cv2.putText(display_frame, label, 
                                              (x1 + 5, y1 - 20), 
                                              cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.font_thickness)
                                    
                                    # Draw confidence
                                    if confidence_label:
                                        cv2.putText(display_frame, confidence_label, 
                                                  (x1 + 5, y1 - 5), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, self.text_color, self.font_thickness)
            
            # Add minimal system info
            queue_text = f"Queue: {self.face_queue.qsize()}"
            cv2.putText(display_frame, queue_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Count images in both folders
            known_count = len([f for f in os.listdir(self.save_folder) if f.endswith('.jpg')])
            unknown_count = len([f for f in os.listdir(self.unknown_folder) if f.endswith('.jpg')]) if os.path.exists(self.unknown_folder) else 0
            
            folder_text = f"Known: {known_count} | Unknown: {unknown_count}"
            cv2.putText(display_frame, folder_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def cleanup_worker(self):
        """Worker thread to handle smart file cleanup (silent mode)"""
        while self.running:
            try:
                cleanup_request = self.cleanup_queue.get(timeout=1)
                face_id = cleanup_request['face_id']
                person_name = cleanup_request.get('person_name', 'UNKNOWN')
                
                if person_name != "UNKNOWN":
                    self.cleanup_known_faces(face_id, person_name)
                else:
                    self.cleanup_unknown_faces(face_id)
                        
            except:
                continue
                
        if self.running:
            self.periodic_cleanup()
    
    def cleanup_known_faces(self, face_id, person_name):
        """Smart cleanup for known/recognized faces (silent)"""
        current_images = [f for f in os.listdir(self.save_folder) if f.endswith('.jpg')]
        image_count = len(current_images)
        
        # Always delete the processed face images after recognition
        if face_id in self.saved_faces:
            files_to_delete = self.saved_faces[face_id]
            self.delete_face_files(files_to_delete, face_id, "known")
        
        # If we have more than max_images_per_folder, delete the oldest ones
        if image_count > self.max_images_per_folder:
            self.cleanup_oldest_files(self.save_folder, "known")
    
    def cleanup_unknown_faces(self, face_id):
        """Smart cleanup for unknown faces (silent)"""
        if not os.path.exists(self.unknown_folder):
            return
            
        current_unknown = [f for f in os.listdir(self.unknown_folder) if f.endswith('.jpg')]
        unknown_count = len(current_unknown)
        
        # Always delete the processed face images after recognition
        if face_id in self.saved_unknown_faces:
            files_to_delete = self.saved_unknown_faces[face_id]
            self.delete_face_files(files_to_delete, face_id, "unknown")
        
        # If we have more than max_images_per_folder, delete the oldest ones
        if unknown_count > self.max_images_per_folder:
            self.cleanup_oldest_files(self.unknown_folder, "unknown")
    
    def delete_face_files(self, file_paths, face_id, face_type):
        """Delete face image files (silent)"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        
        # Remove from tracking
        if face_type == "known" and face_id in self.saved_faces:
            del self.saved_faces[face_id]
        elif face_type == "unknown" and face_id in self.saved_unknown_faces:
            del self.saved_unknown_faces[face_id]
    
    def cleanup_oldest_files(self, folder_path, folder_type):
        """Remove oldest files to keep only max_images_per_folder images"""
        try:
            files_in_folder = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            
            if len(files_in_folder) <= self.max_images_per_folder:
                return
            
            # Get files with creation time
            files_with_time = []
            for filename in files_in_folder:
                filepath = os.path.join(folder_path, filename)
                try:
                    file_time = os.path.getctime(filepath)
                    files_with_time.append((filename, filepath, file_time))
                except:
                    continue
            
            # Sort by creation time (oldest first)
            files_with_time.sort(key=lambda x: x[2])
            
            # Delete oldest files to keep only max_images_per_folder
            files_to_delete = files_with_time[:-self.max_images_per_folder]
            
            for filename, filepath, file_time in files_to_delete:
                try:
                    os.remove(filepath)
                except:
                    pass
                    
        except:
            pass
    
    def update_recognition_results(self):
        """Update recognition results and handle cleanup (silent)"""
        while not self.recognition_results.empty():
            try:
                result = self.recognition_results.get_nowait()
                face_id = result['face_id']
                person_name = result['person_name']
                confidence = result['confidence']
                timestamp = result['timestamp']
                bbox = result['bbox']
                
                # Store recognition result
                self.current_recognitions[face_id] = {
                    'person_name': person_name,
                    'confidence': confidence,
                    'timestamp': timestamp,
                    'bbox': bbox
                }
                
                # Schedule cleanup based on recognition result
                cleanup_request = {
                    'face_id': face_id,
                    'person_name': person_name
                }
                self.cleanup_queue.put(cleanup_request)
                
                # Clean old recognition results (older than 3 seconds)
                current_time = time.time()
                self.current_recognitions = {
                    k: v for k, v in self.current_recognitions.items()
                    if current_time - v['timestamp'] < 3.0
                }
                
            except:
                break
    
    def get_recognition_for_area(self, x1, y1, x2, y2):
        """Get recognition result for a face area"""
        best_match = None
        best_overlap = 0
        
        for face_id, result in self.current_recognitions.items():
            stored_bbox = result['bbox']
            if stored_bbox:
                overlap = self.calculate_bbox_overlap((x1, y1, x2, y2), stored_bbox)
                if overlap > best_overlap and overlap > 0.3:
                    best_overlap = overlap
                    best_match = result
        
        if best_match:
            return best_match['person_name'], best_match['confidence']
        return None, None
    
    def calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0
        
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
        
    def save_face(self, face_img, face_resized):
        """Save detected face to folder (silent)"""
        self.face_counter += 1
        face_id = f"face_{self.face_counter}_{int(time.time())}"
        
        # Save original face image
        filename = f"{face_id}.jpg"
        filepath = os.path.join(self.save_folder, filename)
        cv2.imwrite(filepath, face_img)
        
        # Save resized face image
        filename_resized = f"{face_id}_resized.jpg"
        filepath_resized = os.path.join(self.save_folder, filename_resized)
        cv2.imwrite(filepath_resized, face_resized)
        
        # Track saved files
        self.saved_faces[face_id] = [filepath, filepath_resized]
        
        # Store as numpy array (keep limited amount)
        face_array = {
            'image': face_img,
            'image_resized': face_resized,
            'filename': filename,
            'face_id': face_id,
            'timestamp': time.time()
        }
        self.face_arrays.append(face_array)
        
        # Keep only last 10 numpy arrays in memory
        if len(self.face_arrays) > 10:
            self.face_arrays = self.face_arrays[-10:]
        
        # Save numpy arrays less frequently
        if len(self.face_arrays) % 10 == 0:
            self.save_arrays_to_file()
            
        return face_id
    
    def save_arrays_to_file(self):
        """Save face arrays to pickle file (silent)"""
        arrays_file = os.path.join(self.save_folder, "face_arrays.pkl")
        with open(arrays_file, 'wb') as f:
            pickle.dump(self.face_arrays, f)
    
    def get_face_queue(self):
        return self.face_queue
    
    def get_recognition_results_queue(self):
        return self.recognition_results
    
    def periodic_cleanup(self):
        """Periodic cleanup (silent)"""
        try:
            current_time = time.time()
            self.cleanup_old_files_in_folder(self.save_folder, "known", current_time)
            if os.path.exists(self.unknown_folder):
                self.cleanup_old_files_in_folder(self.unknown_folder, "unknown", current_time)
        except:
            pass
    
    def cleanup_old_files_in_folder(self, folder_path, folder_type, current_time):
        """Cleanup old files in folder (silent)"""
        try:
            files_in_folder = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            
            files_with_time = []
            for filename in files_in_folder:
                filepath = os.path.join(folder_path, filename)
                try:
                    file_time = os.path.getctime(filepath)
                    files_with_time.append((filename, filepath, file_time))
                except:
                    continue
            
            files_with_time.sort(key=lambda x: x[2])
            
            # Keep only max_images_per_folder most recent files
            if len(files_with_time) > self.max_images_per_folder:
                files_to_delete = files_with_time[:-self.max_images_per_folder]
                
                for filename, filepath, file_time in files_to_delete:
                    try:
                        os.remove(filepath)
                    except:
                        pass
        except:
            pass
    
    def stop(self):
        """Stop face detection"""
        self.running = False
        self.save_arrays_to_file()
        self.periodic_cleanup()
    
    def reload_config(self):
        """Reload configuration from config file"""
        if self.config_manager:
            try:
                self.config_manager.reload_config()
                
                # Update configuration values
                self.confidence_threshold = self.config_manager.get('face_detection.confidence_threshold', 0.6)
                self.camera_width = self.config_manager.get('camera.resolution.width', 640)
                self.camera_height = self.config_manager.get('camera.resolution.height', 480)
                self.camera_fps = self.config_manager.get('camera.fps', 30)
                self.max_images_per_folder = self.config_manager.get('face_detection.max_images_per_folder', 10)
                
                # Display settings
                self.window_name = self.config_manager.get('display.window_name', 'Face Recognition')
                self.show_confidence = self.config_manager.get('display.show_confidence', True)
                self.show_bounding_box = self.config_manager.get('display.show_bounding_box', True)
                self.font_scale = self.config_manager.get('display.font_scale', 0.6)
                self.font_thickness = self.config_manager.get('display.font_thickness', 2)
                self.box_thickness = self.config_manager.get('display.box_thickness', 2)
                
                # Colors
                self.known_face_color = tuple(self.config_manager.get('colors.known_face_box', [0, 255, 0]))
                self.unknown_face_color = tuple(self.config_manager.get('colors.unknown_face_box', [0, 0, 255]))
                self.text_bg_color = tuple(self.config_manager.get('colors.text_background', [0, 0, 0]))
                self.text_color = tuple(self.config_manager.get('colors.text_color', [255, 255, 255]))
                
                print("🔄 Configuration reloaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Error reloading configuration: {e}")
                return False
        return False