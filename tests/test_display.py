import pytest
from secora.display import Display
from tqdm import tqdm
from time import sleep

class TrainingProgressMock:
    epoch = 0
    shard = 0
    step = 0


def test_display():
    progress = TrainingProgressMock()

    display = Display(show_progress=True)

    display.set_total(
            4*8*16, 
            3, 
            4, 
            8, 
            16)

    display.start_training()
    display.update(progress)

    for epochs in range(4):
        display.start_epoch()
        for shards in range(8):
            display.start_shard()
            for steps in range(16):
                sleep(0.2)
                progress.epoch = epochs
                progress.shard = shards
                progress.step = steps
                display.update(progress)
            display.start_embedding()
            for steps in range(3):
                sleep(0.2)
                display.update(progress, embedding_step=steps)

    display.close()
