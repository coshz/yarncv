import os
import re
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from utils import make_logger
from dataset import YARN_Dataset
from model import YARN_Model
from config import C


class YarnTrainer:
    def __init__(self, train=True):
        self.model_name_ = C.MODEL_NAME
        self.device_ = self.get_device_()
        self.start_epoch_ = 0
        self.dataloader = self.create_dataloader_(C.BATCH_SIZE, train)
        self.model = self.create_model_(self.model_name_).to(self.device_)
        self.criterion = nn.CrossEntropyLoss().to(self.device_)
        self.optimizer = optim.SGD(self.model.parameters(),lr=C.LEARNING_RATE,momentum=0.9)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.logger = make_logger(os.path.join(C.LOG_DIR, f"train-{self.model_name_}"))

    @staticmethod
    def create_dataloader_(batch_size=4,train=False):
        dataset = YARN_Dataset(C.DATA_DIR, C.DATA_TRAIN if train else C.DATA_TEST)
        if train:
            generator_ = torch.Generator()
            generator_.manual_seed(42)
            train_set, val_set = random_split(
                dataset, 
                [int(len(dataset)*0.9), len(dataset) - int(len(dataset)*0.9)],
                generator_)
            dataloader = {
                'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
                ,'val': DataLoader(val_set,batch_size=batch_size, num_workers=2)
            }
        else: # test
            dataloader = {
                'test': DataLoader(dataset)
            }
        return dataloader
    
    @staticmethod
    def create_model_(model_name):
       return YARN_Model(model_name)
    
    @staticmethod
    def get_device_():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device 

    def train_and_validate(self, num_epochs, val_interval, resume=True):
        start_epoch = 0
        val_score_best = 0.0
        if resume:
            ok, ckpt = self.load_checkpoint_()
            if ok:
                self.logger.info("checkpoint loaded from {}".format(ckpt))
            else:
                self.logger.warning("`resume` is true but no checkpoint is found, start from scratch")
        for epoch in range(start_epoch, num_epochs):
            self.logger.info("Epoch {:2d}/{} Start !!!".format(epoch, num_epochs-1))
            L = len(self.dataloader['train'])
            for i, (IMAGEs, LABELs) in enumerate(self.dataloader['train']):
                IMAGEs = IMAGEs.to(self.device_)
                LABELs = LABELs.to(self.device_)
                OUTPUTs = self.model(IMAGEs)
                loss = self.criterion(OUTPUTs, LABELs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % int(L*0.1) == 0:
                    self.logger.debug("Epoch {:2d} {:5d}/{:5d}: loss = {:.6f}, lr = {:.2e}"
                    .format(epoch, i, L, 
                            loss, 
                            # self.optimizer.param_groups[0]['lr']
                            self.scheduler.get_last_lr()[0]))
            self.scheduler.step()

            if (epoch + 1) % val_interval == 0:
                val_score = self.compute_acc_(self.dataloader['val'], self.model)
                self.logger.info("validation score = {:.2f}".format(val_score))
                if val_score > val_score_best:
                    val_score_best = val_score
                    ckpt_path = f"{self.model_name_}-{val_score_best:.2f}.pth"
                    self.logger.info(f"saving checkpoint to `{ckpt_path}`...")
                    self.save_checkpoint_(epoch,ckpt)
    
    def test(self, ckpt_file=None):
        ok, ckpt = self.load_checkpoint_(ckpt_file)
        if ok:
            self.logger.info("checkpoint loaded from {}".format(ckpt))
            acc = self.compute_acc_(self.dataloader['test'],self.model)
            self.logger.info("Test: acc = {:.2f}".format(acc))
        else:
            raise FileNotFoundError("ckpt file not found: `f{ckpt_file}`")

    def compute_acc_(self, dataloader, model):
        correct = 0
        total = 0
        with torch.no_grad():
            for IMAGEs, LABELs in dataloader:
                IMAGEs, LABELs = IMAGEs.to(self.device_), LABELs.to(self.device_)
                OUTPUTs = model(IMAGEs)
                PREDICTEDs = torch.argmax(OUTPUTs, 1)
                total += LABELs.size(0)
                correct += (PREDICTEDs==LABELs).sum().item()
        acc = 100.0 * correct/total
        return acc
    
    def save_checkpoint_(self, epoch, ckpt_file):
        ckpt_file = os.path.join(C.CKPT_DIR, ckpt_file)
        torch.save({
                'epoch': epoch + 1
                ,'model': self.model.state_dict()
                ,'optimizer': self.optimizer.state_dict()
                ,'scheduler': self.scheduler.state_dict()
            }, 
            ckpt_file)
        
    def load_checkpoint_(self, best=False, ckpt_file=None):
        """load the specified OR best / latest checkpoint"""
        ckpt_regex = rf"{self.model_name_}-(\d+\.\d+).pth"
        if ckpt_file is None:
            if best: 
                best_file, best_acc = None, 0.0
                for file in os.listdir(C.CKPT_DIR):
                    ckpt_match = re.search(ckpt_regex, file)
                    if ckpt_match:
                        acc = float(ckpt_match.group(1))
                        if acc > best_acc:
                            best_acc = acc
                            best_file = os.path.join(C.CKPT_DIR, file)
                ckpt_file = best_file
            else:
                latest_file, latest_time = None, 0
                for file in os.listdir(C.CKPT_DIR):
                    ckpt_match = re.search(ckpt_regex, file)
                    if not ckpt_match: continue
                    file_time = os.path.getmtime(os.path.join(C.CKPT_DIR, file))
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = os.path.join(C.CKPT_DIR, file)
                ckpt_file = latest_file

        if ckpt_file is not None and os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location=self.get_device_())
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.start_epoch_ = ckpt['epoch']
            return True, ckpt_file
        else:
            return False, None