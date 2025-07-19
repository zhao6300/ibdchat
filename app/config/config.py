
import os
import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize config from yaml file"""
        self.config_path = config_path
        self.config_dict = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load config from yaml file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f)
                return config
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse config file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from config by key"""
        return self.config_dict.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get value from config by key using dict-like syntax"""
        return self.config_dict[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config"""
        return key in self.config_dict
