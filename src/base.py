
class TrainerBase:
    def train_and_validate(self, num_epochs, val_interval):
        raise NotImplementedError


class PredicatorBase:
    def __call__(self, img: str | bytes | list[str] | list[bytes]): 
        if isinstance(img, list):
            return self.predict_all(img)
        else:
            return self.predict(img)
    
    def predict(self, img:str|bytes) -> int:
        if isinstance(img, str): 
            return self.predict_from_path(img)
        else:
            return self.predict_from_bytes(img)
    
    def predict_all(self, imgs:list[bytes]|list[str]) -> list[int]:
        return [self.predict(img) for img in imgs]
    
    def predict_from_bytes(self, img:bytes) -> int:
        raise NotImplementedError
    
    def predict_from_path(self, img:str) -> int:
        raise NotImplementedError

    
class TesterBase:
    def __call__(self, img):
        return self.test(img)
    
    def test(self, dataloader, full_metric=False):
        pred = self.get_predicator()
        if pred is None: raise Exception("predicator not set")
        total = len(dataloader)
        n = 0
        for img, label in dataloader:
            if pred(img) == label: n += 1
        acc = n / total
        return acc    
    
    def set_predicator(self, predicator:PredicatorBase):
        raise NotImplementedError
    
    def get_predicator(self) -> PredicatorBase | None:
        raise NotImplementedError
    
    def get_dataloader_from_csv(self, csv):
        raise NotImplementedError
