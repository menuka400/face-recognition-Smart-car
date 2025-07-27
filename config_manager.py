import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML config file: {e}")
        except Exception as e:
            raise Exception(f"Error loading config file: {e}")
    
    def reload_config(self) -> Dict[str, Any]:
        """Reload configuration from file"""
        self.config = self.load_config()
        return self.config
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'face_detection.confidence_threshold')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_face_detection_config(self) -> Dict[str, Any]:
        """Get face detection configuration"""
        return self.config.get('face_detection', {})
    
    def get_face_recognition_config(self) -> Dict[str, Any]:
        """Get face recognition configuration"""
        return self.config.get('face_recognition', {})
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return self.config.get('camera', {})
    
    def get_display_config(self) -> Dict[str, Any]:
        """Get display configuration"""
        return self.config.get('display', {})
    
    def get_colors_config(self) -> Dict[str, Any]:
        """Get colors configuration"""
        return self.config.get('colors', {})
    
    def update_config(self, key_path: str, value: Any):
        """Update configuration value and save to file"""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
        
        # Save to file
        self.save_config()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(self.config, file, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise Exception(f"Error saving config file: {e}")
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self.config[key]
    
    def __contains__(self, key):
        """Allow 'in' operator"""
        return key in self.config
