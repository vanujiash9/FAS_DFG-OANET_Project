import yaml
import os
import torch
import collections

class ConfigLoader:
    def __init__(self, config_paths):
        self.configs = {}
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        for key, path in config_paths.items():
            full_path = os.path.join(os.path.dirname(__file__), path)
            with open(full_path, 'r') as f:
                raw_config = yaml.safe_load(f)
                self.configs[key] = self._resolve_paths(raw_config)
        
        self._create_dirs()

    def _resolve_paths(self, config_dict):
        all_configs_flat = {}
        for k_outer, v_outer in self.configs.items():
            for k_inner, v_inner in v_outer.items():
                all_configs_flat[k_inner] = v_inner
        
        for k, v in config_dict.items():
            all_configs_flat[k] = v

        resolved_config = {}
        for k, v in config_dict.items():
            if isinstance(v, str):
                temp_v = v
                while "${" in temp_v and "}" in temp_v:
                    start = temp_v.find("${")
                    end = temp_v.find("}", start)
                    var_name = temp_v[start+2:end]
                    
                    resolved_value = all_configs_flat.get(var_name)
                    
                    if resolved_value is not None:
                        temp_v = temp_v.replace(f"${{{var_name}}}", str(resolved_value))
                    else:
                        break
                
                if (k.endswith("_DIR") or k.endswith("_PATH")) and not os.path.isabs(temp_v):
                    if k == "BASE_DIR":
                        resolved_config[k] = self.base_dir
                    else:
                        resolved_config[k] = os.path.join(self.base_dir, temp_v)
                else:
                    resolved_config[k] = temp_v
            elif isinstance(v, dict):
                resolved_config[k] = self._resolve_paths(v)
            else:
                resolved_config[k] = v
        return resolved_config
    
    def _create_dirs(self):
        dirs_to_create = set()
        for config_name in self.configs:
            current_cfg = self.configs[config_name]
            for k, v in current_cfg.items():
                if isinstance(v, str):
                    if k.endswith("_DIR"):
                        dirs_to_create.add(v)
                    elif k.endswith("_PATH"):
                        dirs_to_create.add(os.path.dirname(v))
        
        dirs_to_create.add(os.path.join(self.base_dir, self.configs['data']['PROCESSED_DATA_DIR'], 'dfg_real_faces', 'val'))
        dirs_to_create.add(os.path.join(self.base_dir, self.configs['data']['PROCESSED_DATA_DIR'], 'oanet_dataset', 'live'))
        dirs_to_create.add(os.path.join(self.base_dir, self.configs['data']['PROCESSED_DATA_DIR'], 'oanet_dataset', 'spoof'))
        
        for d in sorted(list(dirs_to_create)):
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    def get_config(self, name):
        return self.configs.get(name)

cfg = ConfigLoader({
    "data": "data.yaml",
    "dfg": "dfg.yaml",
    "oanet": "oanet.yaml",
})

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

for key in cfg.configs:
    cfg.configs[key] = AttrDict(cfg.configs[key])
    setattr(cfg, key, cfg.configs[key])

setattr(cfg, "DEVICE", torch.device(cfg.dfg.DEVICE if torch.cuda.is_available() else "cpu"))