import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import datetime
import os
from sklearn.model_selection import StratifiedKFold,train_test_split

import torch.nn.init as init

def reset_parameters(model, init_method='default'):
    if init_method == 'default' and hasattr(model, 'reset_parameters'):
        model.reset_parameters()
        return
    for name, param in model.named_parameters():
        if 'weight' in name:
            if init_method == 'xavier':
                init.xavier_uniform_(param.data)
            elif init_method == 'kaiming':
                init.kaiming_normal_(param.data, nonlinearity='relu')
            elif init_method == 'orthogonal':
                init.orthogonal_(param.data)
        elif 'bias' in name:
            # 偏置项通常初始化为0
            init.constant_(param.data, 0.0)

def train_with_test_evaluation_and_log(model_name, subject, first_epochs, eary_stop_epoch, batch_size, 
                                       device, X_train, Y_train, X_test, Y_test, model, losser, model_savePath, n_classes):
    '''
    The function of the model train with a single training set, 
    evaluate on test set after each epoch, and log the results to a .txt file.

    Args:
        model_name: Model being trained
        subject: Trained subject
        first_epochs: The number of epochs in the training stage
        batch_size: Batch size
        device: Device for model training
        X_train: The train data
        Y_train: The train label
        X_test: The test data
        Y_test: The test label 
        model_savePath: Path to save the model
        n_classes: Number of categories
    '''
    log_write = open(model_savePath + "/log.txt", "w")
    log_write.write('\nTraining on subject ' + str(subject) + '\n')

    # Create train dataset (no validation set, only training data)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long))  
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    info = f"Subject_{subject} training without validation and evaluating on the test set after each epoch."
    print(info)
    log_write.write(info + '\n')
    remaining_epoch = eary_stop_epoch
    # Initialize the model and optimizer
    model.apply(reset_parameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=1e-3)
    # 加入余弦退火学习率调整机制
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=first_epochs)

    # Initialize best test accuracy and model
    best_test_acc = 0
    best_test_loss = np.inf
    best_model = None

    # Training loop
    for epoch in range(first_epochs):
        loss_train = 0
        accuracy_train = 0

        model.train()
        for inputs, target in train_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)

            optimizer.zero_grad()  # Clear gradients
            output = model(inputs)  # Forward pass and compute loss
            loss = losser(output, target)
            loss.backward()  # Backward pass and compute gradients
            optimizer.step()  # Update parameters

            accuracy_train += torch.sum(torch.argmax(output, dim=1) == target, dtype=torch.float32) / len(train_dataset)
            loss_train += loss.detach().item() / len(train_dataloader)
        remaining_epoch = remaining_epoch - 1
        test_accuracy = evaluate_on_test_set(model, X_test, Y_test, device, losser, n_classes)
        test_loss = losser(model(torch.tensor(X_test, dtype=torch.float32).to(device)), torch.tensor(Y_test, dtype=torch.long).to(device)).item()
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            bestbath = epoch 
            best_model = copy.deepcopy(model.state_dict())
        if  test_loss < best_test_loss:
            best_test_loss = test_loss
        # 更新学习率
        scheduler.step()
        info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info = info + '\tEpoch:{0:3}\tTra_Loss:{1:.3f}\tTr_acc:{2:.3f}\tVa_Loss:{3:.3f}\tVa_acc:{4:.3f}\tMinVloss:{5:.3f}\tToacc:{6:.3f}\tMaxVacc:{7:.3f}\tbestbath:{8:.3f}\tramainingEpoch:{9:3}'\
            .format(epoch, loss_train, accuracy_train, test_loss, test_accuracy, loss_train, test_accuracy, best_test_acc, bestbath, remaining_epoch)
        log_write.write(info + '\n')
        print(info)
    # 保存最后一个模型
    last_model = copy.deepcopy(model.state_dict())
    file_name = '{}_sub{}_last.pth'.format(model_name, subject)
    print(f"Saving last model as {file_name}")
    torch.save(last_model, os.path.join(model_savePath, file_name))
    # Log model saving
    info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = info + ' The last model was saved successfully!'
    print(info)
    log_write.write(info + '\n')
    log_write.close()


def validate_model(model, dataset, device, losser, batch_size=512, n_calsses=4):
    loader = DataLoader(dataset, batch_size=batch_size)
    loss_val = 0.0
    accuracy_val = 0.0
    confusion_val = np.zeros((n_calsses,n_calsses), dtype=np.int8)
    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            target = target.to(device)

            probs = model(inputs)
            loss = losser(probs, target)

            loss_val += loss.detach().item()
            accuracy_val += torch.sum(torch.argmax(probs,dim=1) == target, dtype=torch.float32)

            y_true = target.to('cpu').numpy()
            y_pred = probs.argmax(dim=-1).to('cpu').numpy()
        
        loss_val = loss_val / len(loader)
        accuracy_val = accuracy_val / len(dataset)

    return loss_val, accuracy_val



def evaluate_on_test_set(model, X_test, Y_test, device, losser, n_classes):
    '''
    Function to evaluate the model on the test set.
    
    Args:
        model: The trained model
        X_test: Test data
        Y_test: Test labels
        device: Device for model
        losser: Loss function
        n_classes: Number of classes
    
    Returns:
        test_accuracy: The accuracy on the test set
    '''
    model.eval()  # Set the model to evaluation mode
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long))
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    correct = 0
    total = 0
    loss_val = 0

    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, target in test_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)
            loss = losser(output, target)
            loss_val += loss.item()

            pred = torch.argmax(output, dim=1)
            correct += torch.sum(pred == target).item()
            total += target.size(0)

    accuracy = correct / total
    return accuracy