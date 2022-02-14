import unittest
import os
from enum import Enum
from secora.config import *


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

class Color2(Enum):
    RED = 4
    GREEN = 5
    BLUE = 6

class TestConfig(unittest.TestCase):
    def test_create_setting(self):
        a = IntSetting('a number')
        a.set(1)
        print(a.value)

        a = IntOption('a number', 1)
        a.set(1)
        print(a.value)
        a.value

        a = FloatSetting('a number')
        a.set(1.)
        a.value
        print(a.value)

        a = FloatOption('a number', 1.)
        a.set(1.)
        a.value
        print(a.value)


        a = EnumSetting('a enum', Color)
        a.set(Color.RED)
        a.value
        print(a.value)

        a = EnumOption('a number', Color2, Color2.RED)
        a.set(Color2.RED)
        a.value

        print(a.value)
        a = DirectorySetting('a number')
        a.set("/tmp")
        a.value
        print(a.value)

        a = DirectorySetting('a number', "/tmp")
        a.set("/home")
        a.value
        print(a.value)
        #self.assertEqual(1,1)

    def test_config_class(self):
        a = SimpleConfig()
        a.add(IntSetting('hello'))
        a['hello'] = 1
        print(a['hello'])

