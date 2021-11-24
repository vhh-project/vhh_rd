import yaml

class Config(object):
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print("COULD NOT LOAD YAML FILE")
                quit()

    def __getitem__(self, key):
        """
        Overloads [] operators, can be used like config["API_ENDPOINT"]
        """
        return self.config[key]

