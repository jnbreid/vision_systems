from tqdm import tqdm
import torch

from utils.utils import save_model, load_model
from utils.eval import eval_model

class Trainer:

    def __init__(self, model,
                    NUM_EPOCHS,
                    train_loader,
                    valid_loader,
                    EVAL_FREQ,
                    SAVE_FREQ,
                    savepath,
                    writer,
                    lr = 2e-3,
                    gamma = 0.1,
                    step_size = 5,
                    loadpath = None):

        self.model = model
        self.NUM_EPOCHS = NUM_EPOCHS
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.EVAL_FREQ = EVAL_FREQ
        self.SAVE_FREQ = SAVE_FREQ
        self.savepath = savepath
        self.writer = writer
        self.lr = lr
        self.gamma = gamma
        self.step_size = step_size
        self.loadpath = loadpath

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = torch.nn.MSELoss() # is the wanted loss from the task
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = self.step_size)

        self.init_epoch = 0
        self.iter_ = 1
        self.stats = {
            "epoch": [],

            "train_loss" : {},
            "validation_loss" : {},

            "euler_angle" : {},
            "MPJPE" : {},
            "Geodesic" : {},
            "PCK" : {}
            }

        if self.loadpath != None:
            self.model, self.optimizer, self.init_epoch, self.stats, self.iter_ = load_model(self.model, self.optimizer, self.scheduler, loadpath)
            print('loaded model: ' + loadpath)


    def train_one_step(self, items):
        seed_frames = items[0].to(self.device)
        prediction_frames = items[1].to(self.device)

        self.optimizer.zero_grad()

        output = self.model(seed_frames)
        loss = self.criterion(seed_frames, output)

        loss.backward()

        self.optimizer.step()

        return loss.item()


    def train(self):
        self.model.train()
        progress_bar = tqdm(range(self.NUM_EPOCHS), total=self.NUM_EPOCHS, initial=self.init_epoch)
        
        for i in progress_bar:
            epochiter_ = 1
            for item in self.train_loader:

                loss = self.train_one_step(item)

                #if i % 1 == 0:
                progress_bar.set_description(f"Ep {i+1}: Iter {epochiter_}/{len(self.train_loader)}: Loss={round(loss,5)}")

                # adding stuff to tensorboard
                self.writer.add_scalar(f'training_loss', loss, global_step = self.iter_)

                if i % self.EVAL_FREQ == 0:
                    valid_loss, geodesic, eulerangle = eval_model(self.model, self.valid_loader, self.criterion, self.device, metrik = 'angle')
                    self.writer.add_scalar(f'validation_loss', valid_loss, global_step = i)
                    self.writer.add_scalar(f'euler_angle', eulerangle, global_step = i)
                    self.writer.add_scalar(f'geodesic', geodesic, global_step = i)
                
                    self.model.train()

                if i % self.SAVE_FREQ == 0:
                    save_model(model=self.model,
                            optimizer = self.optimizer,
                            epoch = i,
                            stats = self.stats,
                            path=self.savepath,
                            scheduler=self.scheduler,
                            iteration=self.iter_,
                            best=False)
                    #print(f"checkpoint created")



                epochiter_ = epochiter_ + 1
                self.iter_ = self.iter_ + 1

            self.scheduler.step()

        save_model(model=self.model,
                            optimizer = self.optimizer,
                            epoch = i,
                            stats = self.stats,
                            path=self.savepath,
                            scheduler=self.scheduler,
                            iteration=self.iter_,
                            best=False)
        
        return self.stats, self.model