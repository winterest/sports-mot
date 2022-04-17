import torch
import torch.nn.functional as F
from tqdm import tqdm


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {"epoch": epoch, "state_dict": state_dict}
    if not (optimizer is None):
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)


class Trainer(object):
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def run_epoch(self, epoch, data_loader, time_str):
        save_interval = 100
        self.model.train()
        for batch_idx, batch in enumerate(data_loader):
            device = self.model.dummy_param.device
            self.optimizer.zero_grad()

            ref_frame, cur_frame = batch[0].to(device), batch[1].to(device)
            (
                theta_ref_to_cur,
                theta_cur_to_ref,
                ref_mask,
                cur_mask,
            ) = self.model(ref_frame, cur_frame)

            (
                loss_total,
                loss_back,
                loss_fore,
                fore_cur_and_ref,
                fore_ref_and_cur,
            ) = self.loss_fn(
                ref_frame,
                cur_frame,
                ref_mask,
                cur_mask,
                theta_ref_to_cur,
                theta_cur_to_ref,
            )

            loss_total.backward()
            self.optimizer.step()

            with open("./training_{}.log".format(time_str), "a") as file:
                file.write(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n".format(
                        epoch,
                        batch_idx * len(ref_frame),
                        len(data_loader.dataset),
                        100.0 * batch_idx / len(data_loader),
                        loss_total.item(),
                    )
                )

            if batch_idx % 10 == 0:
                print(
                    "fore percentage: {}".format(
                        torch.sum(fore_cur_and_ref)
                        / torch.sum(torch.ones_like(fore_cur_and_ref))
                    )
                )
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_fore: {:.6f}\tLoss_back: {:.6f}".format(
                        epoch,
                        batch_idx * len(ref_frame),
                        len(data_loader.dataset),
                        100.0 * batch_idx / len(data_loader),
                        loss_total.item(),
                        loss_fore.item(),
                        loss_back.item(),
                    )
                )

            if batch_idx % save_interval == 0:
                save_path = "./checkpoints/model_{}_{}.pth".format(
                    epoch, batch_idx
                )
                save_model(save_path, epoch, self.model, self.optimizer)


def train_validate(
    model, train_loader, validation_loader, epoch, optimizer, losses
):
    device = model.device
    model.train()
    epoch_loss, cnt_sample = 0, 0
    for batch_idx, sample in enumerate(train_loader):
        ref, target = sample[:2]
        ref, target = ref.to(device), target.to(device)

        optimizer.zero_grad()

        output0, theta0 = model(ref, target)
        output1, theta1 = model(output0, ref)
        loss_cycle = F.mse_loss(output1, ref)

        loss_trans = F.mse_loss(output0, target)
        loss = loss_trans + loss_cycle
        losses["trans"].append(loss_trans)
        losses["cycle"].append(loss_cycle)

        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCycle: {:.6f}\tTrans: {:.6f}".format(
                    epoch,
                    EPOCHS,
                    batch_idx * len(ref),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    loss_cycle.item(),
                    loss_trans.item(),
                )
            )
            # print(theta)
        epoch_loss += loss * len(sample)
        cnt_sample += len(sample)

    losses["epochs"].append(epoch_loss / cnt_sample)

    valid_loss, cnt_sample = 0, 0
    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(validation_loader):
            ref, target = sample[:2]
            ref, target = ref.to(device), target.to(device)

            output0, theta0 = model(ref, target)
            output1, theta1 = model(output0, ref)
            loss_cycle = F.mse_loss(output1, ref)

            loss_trans = F.mse_loss(output0, target)
            loss = loss_trans + loss_cycle

            valid_loss += loss * len(sample)
            cnt_sample += len(sample)

        losses["validation"].append(valid_loss / cnt_sample)

    print(
        "Training loss: {:.6f} Validation Loss: {:.6f}".format(
            losses["epochs"][-1], losses["validation"][-1]
        )
    )


class AlignTrainer(object):
    """new Trainer for train and validate
    """

    def __init__(self, model, optimizer, loss_fn, opt):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = opt.device
        self.opt = opt

    def run_epoch(self, epoch, data_loader, logger, mode="train"):
        epoch_loop = tqdm(
            enumerate(data_loader), total=len(data_loader), leave=True
        )

        if mode == "train":
            self.model.train()
            epoch_loss, cnt_sample = 0, 0
            for batch_idx, sample in epoch_loop:
                ref, target = sample[:2]
                ref, target = ref.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output0, theta0 = self.model(ref, target)
                output1, theta1 = self.model(output0, ref)
                loss, loss_trans, loss_cycle = self.loss_fn(
                    output0, theta0, output1, theta1, ref, target
                )
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss * len(sample)
                cnt_sample += len(sample)
                epoch_loop.set_description(
                    f"Epoch [{epoch}/{self.opt.epochs}]"
                )
                epoch_loop.set_postfix(
                    loss="{:.6f}\t{:.6f}\t{:.6f}".format(
                        loss_cycle.item(), loss_trans.item(), loss.item()
                    )
                )
                logger.write(
                    "{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCycle: {:.6f}\tTrans: {:.6f}\n".format(
                        mode,
                        epoch,
                        batch_idx * len(ref),
                        len(data_loader.dataset),
                        100.0 * batch_idx / len(data_loader),
                        loss.item(),
                        loss_cycle.item(),
                        loss_trans.item(),
                    )
                )
            logger.write(
                "Train Epoch Loss: {}\n".format(epoch_loss / cnt_sample)
            )
            save_path = "./checkpoints/{}/model_{}_{}.pth".format(
                logger.time_stamp, epoch, batch_idx
            )
            save_model(save_path, epoch, self.model, self.optimizer)

        else:
            self.model.eval()
            epoch_loss, cnt_sample = 0, 0
            with torch.no_grad():
                for batch_idx, sample in epoch_loop:
                    ref, target = sample[:2]
                    ref, target = ref.to(self.device), target.to(self.device)
                    epoch_loss, cnt_sample = 0, 0
                    output0, theta0 = self.model(ref, target)
                    output1, theta1 = self.model(output0, ref)

                    loss, loss_trans, loss_cycle = self.loss_fn(
                        output0, theta0, output1, theta1, ref, target
                    )
                    epoch_loss += loss * len(sample)
                    cnt_sample += len(sample)
                    epoch_loop.set_description(
                        f"Epoch [{epoch}/{self.opt.epochs}]"
                    )
                    epoch_loop.set_postfix(
                        loss="{:.6f}\t{:.6f}\t{:.6f}".format(
                            loss_cycle.item(), loss_trans.item(), loss.item()
                        )
                    )
                    logger.write(
                        "{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCycle: {:.6f}\tTrans: {:.6f}\n".format(
                            mode,
                            epoch,
                            batch_idx * len(ref),
                            len(data_loader.dataset),
                            100.0 * batch_idx / len(data_loader),
                            loss.item(),
                            loss_cycle.item(),
                            loss_trans.item(),
                        )
                    )
                logger.write(
                    "Valid Epoch Loss: {}\n".format(epoch_loss / cnt_sample)
                )
