
import os
import toml
from typing import Any, Dict

class Config:
    def __init__(self, config_path: str = "config.toml"):
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from TOML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config_data = toml.load(f)
        except Exception as e:
            raise Exception(f"Failed to load config file: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from config by key"""
        keys = key.split(".")
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
                
        return value
    
    def __getattr__(self, name: str) -> Any:
        """Allow accessing config values as attributes"""
        return self.get(name)
