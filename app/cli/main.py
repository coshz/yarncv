from app.api import qwen_predictor, local_predictor, make_logger
from app.api.__init__tmp import Grabber

import os
import logging


class YarnRT:
    def __init__(self, predicator):
        self.predictor = predicator 
        self.logger = make_logger('yarn-rt', log_level=logging.INFO, cmd_level=logging.INFO)
        self.grabber = Grabber(logger=self.logger, callback=self.callback)

    def start(self, fps = 30.0, out_dir='img/'): 
        self.grabber.start(fps=fps, out_dir=out_dir)

    def callback(self, img_path):
        label = self.predictor(img_path)
        if label == 0: 
            self.logger.info(f"yarn detected: label = [{label}]")
        else:
            self.logger.warning(f"yarn detected: label = [{label}]")


from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd
import os

class DummyGrabber:
    def __init__(self, callback):
        self.executor_ = ThreadPoolExecutor(max_workers=4)
        self.callback = callback

    def start(self, fps = 30.0, out_dir='img/'):
        test_csv = 'data/data-2/test/yarn-test.test.csv'
        df = pd.read_csv(test_csv, header=0, dtype={'img_path':str,'label_id':int})
        ils = [ (row[0], row[1]) for row in df.itertuples(index=False) ][:20]
        try: 
            i = 0
            while i < len(ils):
                st = time.perf_counter()
                img_path = os.path.abspath('data/data-2/' + ils[i][0])
                if self.callback: 
                    self.executor_.submit(self.callback, img_path)
                et = time.perf_counter()
                time.sleep(max(0, 1/fps + st - et ))
                i += 1
        except KeyboardInterrupt:
            print("stopping by keyboard interrupt...")
        finally:
            self.executor_.shutdown(wait=True)


if __name__ == '__main__':
    out_dir = "img/"
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    # worker = YarnRT(local_predictor)
    worker = YarnRT(qwen_predictor)
    worker.start(fps=2)