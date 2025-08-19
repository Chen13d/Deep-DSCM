import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from collections import OrderedDict

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper
loader, dumper = OrderedYaml()

def parse(opt_path, is_train=True):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=loader)
    return opt

def write2Yaml(data, save_path="test.yaml"):
    with open(save_path, "w") as f:
        yaml.dump(data, f)

if __name__ == "__main__":
    opt_path = r"D:\CQL\codes\microscopy_decouple\options\train_DSRM.yml"
    opt = parse(opt_path=opt_path)
    write2Yaml(opt, save_path=r'D:\CQL\codes\microscopy_decouple\validation\2.19_DSRM_GAN_x1\train_DSRM.yaml')
    