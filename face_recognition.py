import cv2
import numpy as np
import threading
import time
import os
from queue import Queue, Empty
import insightface
from database_manager import DatabaseManager

class FaceRecognizer:
    def __init__(self, model_path, json_database_path, results_queue, config_manager=None):
        self.json_database_path = json_database_path
        self.db_manager = DatabaseManager(json_database_path)
        self.results_queue = results_queue
        self.running = False
        self.config_manager = config_manager
        
        # Load configuration values
        if config_manager:
            self.similarity_threshold = config_manager.get('face_recognition.similarity_threshold', 0.40)
        else:
            self.similarity_threshold = 0.40
        
        # Initialize InsightFace
        try:
            self.app = insightface.app.FaceAnalysis()
            self.app.prepare(ctx_id=0, det_size=(320, 320))
        except Exception as e:
            raise
        
    def recognize_faces(self, face_queue):
        """Main face recognition loop (silent mode)"""
        self.running = True
        
        while self.running:
            try:
                face_data = face_queue.get(timeout=1)
                self.process_face(face_data)
            except Empty:
                continue
            except:
                continue
    
    def process_face(self, face_data):
        """Process a single face for recognition (silent)"""
        try:
            face_img = face_data['image']
            face_id = face_data['face_id']
            bbox = face_data['bbox']
            timestamp = face_data['timestamp']
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Get face embedding
            faces = self.app.get(face_rgb)
            
            if len(faces) > 0:
                face = faces[0]
                embedding = face.embedding
                
                # Search for similar face in JSON database
                similar_person = self.db_manager.find_similar_face(embedding, threshold=self.similarity_threshold)
                
                if similar_person:
                    person_id, person_name, similarity = similar_person
                    confidence_percent = similarity * 100
                    
                    # Send result back to detection thread
                    result = {
                        'face_id': face_id,
                        'person_name': person_name,
                        'confidence': confidence_percent,
                        'timestamp': time.time(),
                        'bbox': bbox
                    }
                    self.results_queue.put(result)
                else:
                    # Send unknown result
                    result = {
                        'face_id': face_id,
                        'person_name': 'UNKNOWN',
                        'confidence': 0.0,
                        'timestamp': time.time(),
                        'bbox': bbox
                    }
                    self.results_queue.put(result)
                    
                    # Handle unknown person
                    self.handle_unknown_person(face_img, face_id)
                        
            else:
                # Send no face result
                result = {
                    'face_id': face_id,
                    'person_name': 'NO_FACE',
                    'confidence': 0.0,
                    'timestamp': time.time(),
                    'bbox': bbox
                }
                self.results_queue.put(result)
                    
        except:
            pass
    
    def handle_unknown_person(self, face_img, face_id):
        """Handle unknown person detection (silent)"""
        unknown_folder = "unknown_faces"
        os.makedirs(unknown_folder, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"unknown_{face_id}_{timestamp}.jpg"
        filepath = os.path.join(unknown_folder, filename)
        cv2.imwrite(filepath, face_img)
        
        # Cleanup old unknown faces to keep only 10 images
        self.cleanup_unknown_folder(unknown_folder)
    
    def cleanup_unknown_folder(self, unknown_folder):
        """Keep only 10 most recent unknown face images"""
        try:
            files_in_folder = [f for f in os.listdir(unknown_folder) if f.endswith('.jpg')]
            
            if len(files_in_folder) <= 10:
                return
            
            # Get files with creation time
            files_with_time = []
            for filename in files_in_folder:
                filepath = os.path.join(unknown_folder, filename)
                try:
                    file_time = os.path.getctime(filepath)
                    files_with_time.append((filename, filepath, file_time))
                except:
                    continue
            
            # Sort by creation time (oldest first)
            files_with_time.sort(key=lambda x: x[2])
            
            # Delete oldest files to keep only 10
            files_to_delete = files_with_time[:-10]
            
            for filename, filepath, file_time in files_to_delete:
                try:
                    os.remove(filepath)
                except:
                    pass
                    
        except:
            pass
    
    def stop(self):
        """Stop face recognition"""
        self.running = False
    
    def reload_config(self):
        """Reload configuration from config file"""
        if self.config_manager:
            try:
                self.config_manager.reload_config()
                self.similarity_threshold = self.config_manager.get('face_recognition.similarity_threshold', 0.40)
                print("🔄 Face recognition configuration reloaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Error reloading face recognition configuration: {e}")
                return False
        return False