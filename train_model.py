import torch
import torch.optim as optim
import copy
import time
from get_optimizer import _get_optimizer

from torch.optim import lr_scheduler


def _train_model(lr, device, model, dataloaders, criterion, opt_type, lr_adjust_mtd,
                 is_lr_adjust, num_epochs, is_inception=False):
    since = time.time()
    # the lambda of lr
    lr_lambda = lambda epoch: 0.95 ** epoch
    exp_lr_scheduler, optimizer_ft = _get_optimizer(model, lr, opt_type, is_lr_adjust, lr_adjust_mtd, lr_lambda)
    # optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # initialize the parameters used to save the best variables
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 1
    # save the picture that can't classified correctly
    f_imgs_val = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if(is_lr_adjust):
                    exp_lr_scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            base_num = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                is_goal = preds == labels.data
                running_corrects += torch.sum(is_goal)
                if phase == 'val':
                    # save the wrongly classified images
                    isnot_goal = 1 - is_goal
                    isnot_goal_pos = torch.nonzero(isnot_goal)
                    isnot_goal_pos = isnot_goal_pos.view(isnot_goal_pos.shape[0])
                    f_conf_batch = outputs[isnot_goal_pos, ...]
                    if(base_num == 0):
                        f_conf_tmp = f_conf_batch
                        f_imgs_idx = isnot_goal_pos
                    else:
                        f_conf_tmp = torch.cat((f_conf_tmp, f_conf_batch), 0)
                        isnot_goal_pos = base_num + isnot_goal_pos
                        f_imgs_idx = torch.cat((f_imgs_idx, isnot_goal_pos), 0)
                    base_num = base_num + len(labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if(is_lr_adjust):
                epoch_lr = exp_lr_scheduler.get_lr()
                epoch_lr = epoch_lr[0]
            else:
                epoch_lr = lr
            output_arr = torch.Tensor([[float(epoch_acc), float(epoch_loss), float(epoch_lr)]])
            print('{} lr: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format(phase, float(epoch_lr), epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'train':
                if epoch == 0:
                    output_arr_train = output_arr
                else:
                    output_arr_train = torch.cat((output_arr_train, output_arr), 0)
            if phase == 'val':
                if epoch == 0:
                    output_arr_val = output_arr
                else:
                    output_arr_val = torch.cat((output_arr_val, output_arr), 0)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                f_imgs_val = {'image_udx': f_imgs_idx, 'conf': f_conf_tmp}

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, float(best_acc), best_epoch, output_arr_val, output_arr_train, f_imgs_val
