import threading
import time
import os
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from config_manager import ConfigManager

def main():
    try:
        # Load configuration
        config = ConfigManager("config.yaml")
        
        # Get configuration values
        face_detection_config = config.get_face_detection_config()
        face_recognition_config = config.get_face_recognition_config()
        
        YOLO_MODEL_PATH = face_detection_config.get('model_path', 'yolov11m-face.pt')
        SAVE_FOLDER = face_detection_config.get('save_folder', 'Face_imagers')
        INSIGHTFACE_MODEL_PATH = face_recognition_config.get('insightface_model_path', 'insightface_models/models/buffalo_l.zip')
        JSON_DATABASE_PATH = face_recognition_config.get('json_database_path', 'face_embeddings.json')
        
        print("🚀 Starting Face Recognition System...")
        print(f"📁 Configuration loaded from: config.yaml")
        
        if not os.path.exists(JSON_DATABASE_PATH):
            print(f"❌ JSON database not found: {JSON_DATABASE_PATH}")
            return
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    try:
        # Initialize components with configuration
        face_detector = FaceDetector(YOLO_MODEL_PATH, SAVE_FOLDER, config)
        
        # Get queues
        face_queue = face_detector.get_face_queue()
        results_queue = face_detector.get_recognition_results_queue()
        
        # Initialize face recognizer with configuration
        face_recognizer = FaceRecognizer(INSIGHTFACE_MODEL_PATH, JSON_DATABASE_PATH, results_queue, config)
        
        # Create threads
        detection_thread = threading.Thread(target=face_detector.detect_faces, name="FaceDetectionThread")
        recognition_thread = threading.Thread(target=face_recognizer.recognize_faces, args=(face_queue,), name="FaceRecognitionThread")
        
        # Start threads
        detection_thread.start()
        recognition_thread.start()
        
        # Configuration monitoring
        config_file_path = "config.yaml"
        config_last_modified = os.path.getmtime(config_file_path) if os.path.exists(config_file_path) else 0
        
        def monitor_config_changes():
            """Monitor config file for changes and reload if needed"""
            nonlocal config_last_modified
            while detection_thread.is_alive() and recognition_thread.is_alive():
                try:
                    if os.path.exists(config_file_path):
                        current_modified = os.path.getmtime(config_file_path)
                        if current_modified > config_last_modified:
                            print("📝 Config file changed, reloading...")
                            if face_detector.reload_config():
                                face_recognizer.reload_config()
                            config_last_modified = current_modified
                    time.sleep(2)  # Check every 2 seconds
                except Exception as e:
                    print(f"❌ Error monitoring config: {e}")
                    time.sleep(5)
        
        # Start config monitoring thread
        config_monitor_thread = threading.Thread(target=monitor_config_changes, name="ConfigMonitorThread", daemon=True)
        config_monitor_thread.start()
        
        print("✅ System running - Press 'q' in video window to quit")
        print("📝 Configuration will be reloaded automatically when config.yaml is modified")
        
        # Keep main thread alive
        while detection_thread.is_alive() and recognition_thread.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Stop components
        if 'face_detector' in locals():
            face_detector.stop()
        if 'face_recognizer' in locals():
            face_recognizer.stop()
        
        # Wait for threads to finish
        if 'detection_thread' in locals() and detection_thread.is_alive():
            detection_thread.join(timeout=5)
        if 'recognition_thread' in locals() and recognition_thread.is_alive():
            recognition_thread.join(timeout=5)
        
        print("✅ System shutdown complete.")

if __name__ == "__main__":
    main()