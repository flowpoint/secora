import pytest
import os
from enum import Enum
from secora.config import *
import tempfile


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

class Color2(Enum):
    RED = 4
    GREEN = 5
    BLUE = 6

test_setting_names = [str(x) for x in range(100)]
test_setting_classes = [ IntSetting, FloatSetting, DirectorySetting ]
test_option_classes = [ IntOption, FloatOption, DirectoryOption ]
test_values = [1, 1., '/tmp']

class TestConfig:
    @pytest.mark.parametrize('setting_class,name,value', zip(test_setting_classes, test_setting_names, test_values))
    def test_setting_creation(self, setting_class, name, value):
        a = setting_class(name)
        a.set(value)
        assert a.value == value

    @pytest.mark.parametrize('setting_class,name,value,default', zip(test_option_classes, test_setting_names, test_values, test_values))
    def test_option_creation(self, setting_class, name, value, default):
        a = setting_class(name, default)
        a.set(value)
        assert a.value == value

    @pytest.mark.parametrize('setting_class,name,value,default', zip(test_option_classes, test_setting_names, test_values, test_values))
    def test_option_mismatch(self, setting_class, name, value, default):
        with pytest.raises(Exception):
            a = setting_class(name, None)

        with pytest.raises(Exception):
            a = setting_class(name, type(type))
            a.set(value)

    def test_enum_setting(self):
        a = EnumSetting('a enum', Color)
        a.set(Color.RED)
        assert a.value == Color.RED

    def test_enum_default_mismatch(self):
        a = EnumSetting('a enum', Color)
        a.set(Color.RED)
        assert a.value == Color.RED

        with pytest.raises(Exception):
            a = EnumOption('a number', Color, Color2.RED)

        a = EnumOption('a number', Color2, Color2.RED)

        with pytest.raises(Exception):
            a.set(Color.RED)
        assert a.value == Color2.RED

    def test_simpleconfig_usage(self):
        a = SimpleConfig()
        a.add(IntSetting('hello'))
        with pytest.raises(Exception):
            b = a['hello']

        with pytest.raises(Exception):
            a['hello'] = 1.

        a['hello'] = 1

        assert a['hello'] == 1

    def test_simpleconfig_compose(self):
        a = SimpleConfig()
        a.add(IntSetting('hello'))

        b = SimpleConfig()
        b.add(IntSetting('hello'))

        c = SimpleConfig()
        c.add(IntSetting('hello2'))

        with pytest.raises(Exception):
            a.compose(b)
        with pytest.raises(Exception):
            b.compose(a)

        ac = a.compose(c)
        print(ac._settings)
        assert 'hello' in ac._settings
        assert 'hello2' in ac._settings
        ca = c.compose(a)
        assert 'hello' in ca._settings
        assert 'hello2' in ca._settings

