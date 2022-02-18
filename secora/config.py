import decimal
import torch
import yaml
import os

from abc import abstractmethod
from abc import ABC
from functools import wraps

def check_config(config):
    if not isinstance(config, dict):
        raise ValueError("the passed object is not a dict")


def load_config(path):
    ''' this verifies and translates the config yaml file 
    to a valid training setup
    '''

    with open(path, 'r') as f:
        yconfig = yaml.safe_load(f)

    check_config(yconfig)

    config = dict()
    config.update(yconfig)
    config['learning_rate'] = float(decimal.Decimal(yconfig['learning_rate']))
    config['optimizer'] = yconfig['optimizer']

    if yconfig['num_gpus'] == 'auto':
        config['num_gpus'] = int(torch.cuda.device_count())
    elif torch.cuda.device_count() >= int(yconfig['num_gpus']) >= 0 and torch.cuda.is_available():
        pass
    else:
        raise ValueError('requested num_gpus not available')


    if not 'checkpoint_dir' in yconfig or yconfig['checkpoint_dir'] == "":
        raise ValueError('checkpoint dir must be specified')

    if not 'logdir' in yconfig or yconfig['logdir'] == "":
        raise ValueError('checkpoint dir must be specified')

    #if yconfig['precision'] == 'mixed' and int(config['num_gpus']) == 0:
    #    raise RuntimeError('cant use cuda amp mixed on cpu')

    if not isinstance(config['cuda_graphs'], bool):
        raise RuntimeError('cuda_graphs has to be bool')


    return config


def overwrite_config(args, config):
    if args.run_name is not None and args.run_name != "":
        config['name'] = args.run_name

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    check_config(config)

    return config

def save_config(config, path):
    with open(path, 'w') as f:
        f.write(yaml.dump(config))


class ConfigValueError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ConfigTypeError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
##

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
    return wrapper

class Setting(ABC):
    def __init__(self, name, *args, **kwargs):
        if not isinstance(name, str):
            raise TypeError(f'argument name has to be a str')

        self._name = name
        self._value = None
        self._is_set = False

    @property
    def name(self):
        return self._name

    def parse(self, s):
        t = self.allowed_type
        return t(str(s))

    @property
    @abstractmethod
    def allowed_type(self):
        pass

    @property
    def value(self):
        if not self._is_set:
            raise RuntimeError(f'setting: {self._name} value must be set before reading')
        return self._value

    def final(self):
        self._is_set = True

    def set(self, val):
        ''' settings are intended to be immutable for the remaining code '''

        if self._is_set == True:
            raise RuntimeError(f'Setting: {self.name} already set')
        if not self.check_type(val) == True:
            raise RuntimeError(f'Setting: {self.name} already set')
        if not self.check(val) == True:
            raise ValueError(f'trying to set setting: {self._name} to prohibited value of: {val}')

        self.final()
        self._value = val

    def check_type(self, val) -> bool:
        if type(val) != self.allowed_type:
            raise TypeError(f'{val} is expected to be of type: {self.allowed_type}, but is {type(val)}')

        if not isinstance(val, self.allowed_type):
            raise ValueError(f'{val} has to be a type but is {self.allowed_type}')
        ''' to be implemented by subclasses '''
        return True

    @abstractmethod
    def check(self, val) -> bool:
        ''' to be implemented by subclasses '''
        raise NotImplementedError()

    #def __repr__(self):
    # return f"Setting('{self._name}')"

    def __repr__(self):
        return f'{{"{self._name}": "{self._value}"}}'

    def __str__(self):
        return f'{{"{self._name}": "{self._value}"}}'



class Option(Setting):
    ''' an optional setting with default value '''
    def __init__(self, name, default, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.check_type(default)
        self.check(default)
        self._default = default

    @property
    def value(self):
        ''' first time the value is requested, and if not overwritten,
        the default will be permanently set '''
        if not self._is_set:
            self.set(self._default)
        return super().value

class BoolSetting(Setting):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def parse(self, s):
        if isinstance(s, str):
            return s == 'False'
        if isinstance(s, bool):
            return s

    @property
    def allowed_type(self):
        return bool

    def check(self, val):
        return True
    

class BoolOption(Option):
    def __init__(self, name, default, *args, **kwargs):
        super().__init__(name, default, *args, **kwargs)

    @property
    def allowed_type(self):
        return bool

    def check(self, val):
        return True
##
class IntSetting(Setting):
    def __init__(self, name, lb=None, ub=None, *args, **kwargs):
        self._lb = lb
        self._ub = ub
        super().__init__(name, *args, **kwargs)

    @property
    def allowed_type(self):
        return int

    def check(self, val):
        if self._lb is not None and val < self._lb:
            return False
        if self._ub is not None and val >= self._ub:
            return False
        return True

    def __repr__(self):
        return f"IntSetting('{self._name}')"


class IntOption(Option):
    def __init__(self, name, default, lb=None, ub=None, *args, **kwargs):
        self._lb = lb
        self._ub = ub
        super().__init__(name, default, *args, **kwargs)

    @property
    def allowed_type(self):
        return int

    def check(self, val):
        if self._lb is not None and val < self._lb:
            return False
        if self._ub is not None and val >= self._ub:
            return False
        return True


class FloatSetting(Setting):
    def __init__(self, name, lb=None, ub=None, *args, **kwargs):
        self._lb = lb
        self._ub = ub
        super().__init__(name, *args, **kwargs)

    @property
    def allowed_type(self):
        return float

    def check(self, val):
        if self._lb is not None and val < self._lb:
            return False
        if self._ub is not None and val >= self._ub:
            return False
        return True

class FloatOption(Option):
    def __init__(self, name, default, lb=None, ub=None, *args, **kwargs):
        self._lb = lb
        self._ub = ub
        super().__init__(name, default, *args, **kwargs)

    @property
    def allowed_type(self):
        return float

    def check(self, val):
        if self._lb is not None and val < self._lb:
            return False
        if self._ub is not None and val >= self._ub:
            return False
        return True


class EnumSetting(Setting):
    def __init__(self, name, enum, *args, **kwargs):
        self.enum = enum
        super().__init__(name, *args, **kwargs)

    def parse(self, s):
        return self.enum(s)

    @property
    def allowed_type(self):
        return self.enum

    def check(self, val):
        return True

class EnumOption(Option):
    def __init__(self, name, enum, default, lb=None, ub=None, *args, **kwargs):
        self.enum = enum
        super().__init__(name, default, *args, **kwargs)

    @property
    def allowed_type(self):
        return self.enum

    def check(self, val):
        return True


class DirectorySetting(Setting):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    @property
    def allowed_type(self):
        return str

    def check(self, val):
        if not os.path.isdir(val):
            raise ValueError('value has to be an existing directory')
        return True


class DirectoryOption(Option):
    def __init__(self, name, default, lb=None, ub=None, *args, **kwargs):
        super().__init__(name, default, *args, **kwargs)

    @property
    def allowed_type(self):
        return str

    def check(self, val):
        if not os.path.isdir(val):
            raise ValueError('value has to be an existing directory')
        return True


class FileSetting(Setting):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    @property
    def allowed_type(self):
        return str

    def check(self, val):
        if not os.path.isfile(val):
            raise ValueError('value has to be an existing file')
        return True


class FileOption(Option):
    def __init__(self, name, default, lb=None, ub=None, *args, **kwargs):
        super().__init__(name, default, *args, **kwargs)

    @property
    def allowed_type(self):
        return str

    def check(self, val):
        if not os.path.isdir(val):
            raise ValueError('value has to be an existing file')
        return True


class Config(ABC):
    @abstractmethod
    def add(self, setting):
        pass

    @abstractmethod
    def __setitem__(self, name, value):
        pass

    @abstractmethod
    def __getitem__(self, name):
        pass

    @abstractmethod
    def check(self):
        pass


class SimpleConfig:
    def __init__(self):
        self._settings = {}

    def add(self, setting):
        if not isinstance(setting, Setting):
            raise TypeError("setting has to be of type config.Setting")
        self._settings[setting.name] = setting

    def __setitem__(self, name, val):
        return self._settings[name].set(val)

    def __getitem__(self, name):
        return self._settings[name].value

    def check(self) -> bool:
        for k, v in self._settings.items():
            v.check(v.value)

    def to_dict(self):
        d = dict()
        for k, v in self._settings.items():
            d[k] = v.value

        return d


    def final(self):
        for v in self._settings.values():
            v.final()

    @property
    def settings(self):
        return self._settings

    def compose(self, config2):
        nconf = SimpleConfig()
        for k,v in self._settings.items():
            if k in config2._settings:
                raise RuntimeError("can't compose configs, where a setting is overwritten")

            nconf.add(v)
        for k,v in config2._settings.items():
            if k in self._settings:
                raise RuntimeError("can't compose configs, where a setting is overwritten")

            nconf.add(v)

        return nconf

    def overwrite(self, config_candidate):
        for k,v in config_candidate.items():
            config[k] = config.settings[k].parse(v)




if __name__ == "__main__":
    desc = 'config utility, by default, checks config validity'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        yconfig = yaml.safe_load(f)

    check_config(yconfig)
    print('config seems valid')
