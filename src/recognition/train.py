import os
import torch
from torch import nn
import torch.optim as optim
from .config import C
from .utils import make_logger, get_device, search_checkpoint, img_augment
from .metrics import eval_acc, eval_metrics


class YarnTrainer:
    def __init__(self, model, dataloaders):
        self.device_ = get_device()
        self.start_epoch_ = 0
        self.dataloaders = dataloaders
        self.model = model.to(self.device_)
        self.criterion = nn.CrossEntropyLoss().to(self.device_)
        self.optimizer = optim.SGD(self.model.parameters(),lr=C.LEARNING_RATE,momentum=0.9,weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.logger = make_logger(os.path.join(C.LOG_DIR, f"train-{C.MODEL_NAME}"))

    def train_and_validate(self, num_epochs, val_interval, resume=True):
        start_epoch = 0
        val_score_best = 0.0
        if resume:
            ckpt = search_checkpoint(C.MODEL_NAME, C.CKPT_DIR, best=False)
            if ckpt:
                self.load_checkpoint_(ckpt)
                self.logger.info("checkpoint loaded from {}".format(ckpt))
            else:
                self.logger.warning("`resume` is true but no checkpoint is found, start from scratch")
        for epoch in range(start_epoch, num_epochs):
            self.logger.info("Epoch {:2d}/{} Start !!!".format(epoch, num_epochs-1))
            L = len(self.dataloaders['train'])
            for i, (IMAGEs, LABELs) in enumerate(self.dataloaders['train']):
                IMAGEs = IMAGEs.to(self.device_)
                LABELs = LABELs.to(self.device_)
                
                # augmentation 
                IMAGEs = img_augment()(IMAGEs)

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
                val_score = eval_acc(self.dataloaders['val'], self.model)
                self.logger.info("validation score = {:.2f}".format(val_score))
                if val_score > val_score_best:
                    val_score_best = val_score
                    ckpt_file = f"{C.MODEL_NAME}-{val_score_best:.2f}.pth"
                    self.logger.info(f"saving checkpoint to `{ckpt_file}`...")
                    self.save_checkpoint_(epoch, ckpt_file)
    
    def test(self, ckpt_path='', full_metric=False):
        ckpt = ckpt_path or search_checkpoint(C.MODEL_NAME, C.CKPT_DIR, best=True)
        if ckpt:
            self.load_checkpoint_(ckpt)
            self.logger.info("checkpoint loaded from {}".format(ckpt))
        else:
            raise FileNotFoundError(f"checkpoint not found")
        
        torch.backends.cudnn.deterministic = True
        self.model.eval()

        self.logger.info(f"--------------------------------") 
        if not full_metric:
            acc = eval_acc(self.dataloaders['test'],self.model)
            self.logger.info("Test: acc = {:.2f}".format(acc))
        else: 
            metrics = eval_metrics(self.dataloaders['test'], self.model)
            self.logger.info(f"Test: ")
            for k, v in metrics.items():
                self.logger.info(f"-- {k:12}: {v}")
        self.logger.info(f"--------------------------------")
    
    def save_checkpoint_(self, epoch, ckpt_file):
        ckpt_file = os.path.join(C.CKPT_DIR, ckpt_file)
        torch.save({
            'epoch': epoch + 1
            ,'model': self.model.state_dict()
            ,'optimizer': self.optimizer.state_dict()
            ,'scheduler': self.scheduler.state_dict()
        }, ckpt_file)
   
    def load_checkpoint_(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file `{ckpt_path}` not found.")
        ckpt = torch.load(ckpt_path, map_location=self.device_)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch_ = ckpt['epoch'] 