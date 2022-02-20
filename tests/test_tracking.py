import pytest
import os
from secora.tracking import StateTracker
import tempfile

class MockModel:
    def __init__(self):
        self.state = {'key1':"value1"}
    def state_dict(self):
        return self.state
    def load_state_dict(self, d):
        self.state = d

class MockLogger:
    def debug(self,*args, **kwargs):
        pass
    def info(self,*args, **kwargs):
        pass
    def warning(self,*args, **kwargs):
        pass
    def exception(self,*args, **kwargs):
        pass

class TestTracker:
    logger = MockLogger()

    def test_checkpoint_count(self):
        model = MockModel()
        for i in [0,1,10]:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tracker = StateTracker(name='model', logdir=tmpdirname, max_checkpoints=i, model=model, logger=self.logger)

                path = os.path.join(tmpdirname, 'model')

                for k in range(0, i):
                    files = os.listdir(path)
                    assert files == tracker._list_checkpoints()
                    tracker.save()

                for k2 in range(0,10):
                    files = os.listdir(path)
                    tracker.save()
                    assert len(files) <= i

    def test_restore(self):
        model = MockModel()
        with tempfile.TemporaryDirectory() as tmpdirname:
            tracker = StateTracker(name='model', logdir=tmpdirname, max_checkpoints=1, model=model, logger=self.logger)
            
            path = os.path.join(tmpdirname, 'model')
            assert os.listdir(path) == []

            model.state['key1'] = "value2"
            tracker.save()

            model2 = MockModel()
            tracker2 = StateTracker(name='model', logdir=tmpdirname, max_checkpoints=1, model=model2, logger=self.logger)
            tracker2.load_latest()
            assert model2.state == model.state
            assert tracker['model'].state == model.state
    

