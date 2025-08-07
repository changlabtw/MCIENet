import yaml


def search_random_arg(config_path:str, test_case:dict):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


config_path = "conf/MCIENet/gm12878_1000bp_best.yaml"
test_case = {
    
}

search_random_arg(config_path, test_case)