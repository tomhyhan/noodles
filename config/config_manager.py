import yaml
from easydict import EasyDict
import os

class ConfigManager:
    def __init__(self, config_path="./config.yml"):
        with open(config_path, 'r') as yml_file:
            self.config = EasyDict(yaml.safe_load(yml_file))
            
    def get(self, key, default=None):
        return self.config.get(key, default)
        
    def __getitem__(self, key):
        return self.config[key]
    
    def __repr__(self):
        return repr(self.config)
    
if __name__ == "__main__":
    m = ConfigManager()
    print(m.config.swin.train_args.num_epochs)