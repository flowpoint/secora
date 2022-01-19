import unittest
import os
import sys
from secora.tracking import StateTracker
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MockModel:
    def __init__(self):
        self.state = {'key1':"value1"}
    def state_dict(self):
        return self.state
    def load_state_dict(self, d):
        self.state = d


class TestTracker(unittest.TestCase):
    def test_checkpoint_count(self):
        model = MockModel()
        for i in [0,1,10]:
            with tempfile.TemporaryDirectory() as tmpdirname:
                config = {
                    "max_checkpoints": i,
                    'checkpoint_dir': tmpdirname,
                    'name': 'model'
                        }

                tracker = StateTracker(config, model=model)

                path = os.path.join(tmpdirname, 'model')
                self.assertTrue(os.listdir(path) == [])

                for k in range(i+10):
                    files = os.listdir(path)
                    self.assertCountEqual(files, tracker._list_checkpoints())
                    tracker.save()
                    self.assertLessEqual(len(files), k)

    def test_restore(self):
        model = MockModel()
        with tempfile.TemporaryDirectory() as tmpdirname:
            config = {
                "max_checkpoints": 1,
                'checkpoint_dir': tmpdirname,
                'name': 'model'
                }

            tracker = StateTracker(config, model=model)
            
            path = os.path.join(tmpdirname, 'model')
            self.assertTrue(os.listdir(path) == [])

            model.state['key1'] = "value2"
            tracker.save()

            model2 = MockModel()
            tracker2 = StateTracker(config, model=model2)
            tracker2.load_latest()
            self.assertEqual(model2.state, model.state)
            self.assertEqual(tracker['model'].state, model.state)
    

