from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
import yaml

class CardConfig(BaseModel):
    name: str
    issuer: str
    active: bool = True

class NotificationConfig(BaseModel):
    email: Optional[dict[str, List[str] | int]] = None

class MonitoringConfig(BaseModel):
    schedule: str
    cards: List[CardConfig]

class MonitoringSettings(BaseModel):
    monitoring: MonitoringConfig
    notification: NotificationConfig

def load_monitoring_settings(config_path: str = None) -> MonitoringSettings:
    """Load monitoring configuration from YAML file."""
    if config_path is None:
        # Try common locations
        possible_paths = [
            Path("src/agent/config/settings/monitoring.yaml"),
            Path("agent/config/settings/monitoring.yaml"),
            Path("config/settings/monitoring.yaml"),
            Path("config/monitoring.yaml"),
        ]
        
        for path in possible_paths:
            print(f"Trying path: {path} (exists: {path.exists()})")
            if path.exists():
                config_path = str(path)
                break
        else:
            raise FileNotFoundError(
                "Could not find monitoring.yaml in any of these locations: "
                f"{', '.join(str(p) for p in possible_paths)}"
            )
    
    print(f"Loading config from: {config_path}")
    with open(config_path) as f:
        content = f.read()
        print(f"File content: {content}")
        config_dict = yaml.safe_load(content)
        if config_dict is None:
            raise ValueError(f"Empty or invalid YAML file: {config_path}")
    
    return MonitoringSettings(**config_dict) 