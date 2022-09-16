import torch
from sonar_loader import *
from sklearn.model_selection import KFold
from model import *
from eval import *
import tqdm


def train_function(data, model, optimizer, loss_function, scheduler, device):
    model.train()
    epoch_loss = 0

    for index, sample_batch in enumerate(tqdm.tqdm(data)):
        imgs = sample_batch['image']
        gt_mask = sample_batch['mask']

        imgs = imgs.to(device)
        true_masks = gt_mask.to(device)

        outputs = model(imgs)

        # prediction vis
        probs = torch.softmax(outputs, dim=1)
        masks_pred = torch.argmax(probs, dim=1)

        loss = loss_function(outputs, true_masks)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch finished ! Loss: {epoch_loss / index:.4f}, lr:{scheduler.get_last_lr()}')

def validation_epoch(model, val_loader, num_class, device, epoch):
    class_iou, mean_iou = eval_net_loader(model, val_loader, num_class, device, epoch)
    print('Class IoU:', ' '.join(f'{x:.4f}' for x in class_iou), f'  |  Mean IoU: {mean_iou:.4f}')

    return mean_iou

def main(mode='', gpu_id=0, num_epoch=31, train_batch_size=2, test_batch_size=1, classes=[], pretrained=False, save_path=''):
    num_val = 200
    fold_num = 4
    lr = 0.001
    save_term = 5

    dir_checkpoint = f'./checkpoints/UNet_b{train_batch_size}'

    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    data_path = './data/segmentation/'

    total_dataset = sonarDataset(data_path, classes)

    total_len = len(list(natsort.natsorted(os.listdir(os.path.join(data_path, "Images")))))
    total_len = list(range(0,total_len))
    kfold = KFold(n_splits=fold_num, shuffle=False)

    for fold, (train_idx, test_ids) in enumerate(kfold.split(total_len)):
        if fold != 0:
            break
        dataset = torch.utils.data.Subset(total_dataset, train_idx[num_val:])
        dataset_val = torch.utils.data.Subset(total_dataset, train_idx[:num_val])
        dataset_test = torch.utils.data.Subset(total_dataset, test_ids)


        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=train_batch_size, shuffle=True, num_workers=0
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=test_batch_size, shuffle=True, num_workers=0
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=0
        )

        model = UNet(in_channels=1, n_classes=len(classes)).to(device).train()
        # print(model)


        if 'train' in mode:

            if pretrained:
                pre_path = ''
                model.load_state_dict(torch.load(pre_path))
                print('Model loaded from {}'.format(pre_path))

            print('Starting training: '
                  'Epochs: {num_epoch}'
                  'Batch size: {train_batch_size}'
                  'Learning rate: {lr}'
                  'Training size: {len(data_loader.dataset)}'
                  'Device: {str(device)}')

            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr)


            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.3 * epochs), gamma=0.1)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

            loss_function = torch.nn.CrossEntropyLoss()

            for epoch in range(num_epoch):
                print('*** Starting epoch {}/{}. ***'.format(epoch, num_epoch))

                train_function(data_loader, model, optimizer, loss_function, lr_scheduler, device)
                lr_scheduler.step()

                validation_epoch(model, data_loader_val, len(classes), device, epoch)

                if epoch % save_term == 0:
                    state_dict = model.state_dict()
                    if device == "cuda":
                        state_dict = model.module.state_dict()
                    torch.save(state_dict, dir_checkpoint + f'_e_{epoch}.pth')
                    print('Checkpoint epoch: {} saved !'.format(epoch))

                print('****************************\n\n')




if __name__ =="__main__":

    # TODO
    '''
    1. Iou equation check 
    2. visualization (12 color) - complete
    3. test    
    '''

    CLASSES = ['background', 'bottle', 'can', 'chain',
               'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
               'standing-bottle', 'tire', 'valve', 'wall']


    main(mode='train', gpu_id=0, num_epoch=31,
         train_batch_size=16, test_batch_size=1, classes=CLASSES,
         pretrained=False, save_path='')