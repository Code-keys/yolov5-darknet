import yaml
import os


def write_cfg(cfg_file, module_defs):
    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key != 'type':
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file


if __name__ == "__main__":
    yamlf = ''
    yaml_ = yaml.load(yamlf, Loader=yaml.FullLoader)  # model dict
    
    # to do