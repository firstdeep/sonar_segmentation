import cv2
import os
import numpy as np
import natsort
import matplotlib.pyplot as plt

path = './data/segmentation/'
mask_path = os.path.join(path, 'Masks')
image_path = os.path.join(path, 'Images')
save_path = os.path.join(path, 'processing')
sum_path = os.path.join(path, 'img_mask_sum')

def mask_processing():
    '''
    :return: mask image processing and concatenate original images
    '''
    print(len(natsort.natsorted(os.listdir(mask_path))))

    label_list= [0]*12
    pixel_list= [0]*12

    for i in natsort.natsorted(os.listdir(mask_path)):
        print(i)
        mask = cv2.imread(os.path.join(mask_path, i), cv2.IMREAD_GRAYSCALE)
        mask_uniq = np.unique(mask.flatten()) # 0 ~ 11 class
        mask_flat = list(mask.flatten())
        for i in mask_uniq: # number of class
            label_list[int(i)] = label_list[int(i)] + 1
            pixel_list[int(i)] = pixel_list[int(i)] + mask_flat.count(int(i))
    label_list = np.array(label_list)
    pixel_list = np.array(pixel_list)

    np.save('./etc/label.npy', label_list)
    np.save('./etc/pixel.npy', pixel_list)

    return label_list, pixel_list

def draw_plot(label_list, pixel_list):
    indx = [i for i in range(1,12)]
    label_list = label_list[1:]
    pixel_list = pixel_list[1:]

    plt.subplot(2,1,1)
    plt.bar(indx, label_list)
    plt.xlim(0, 12)
    plt.ylim(0, 1000)


    plt.subplot(2, 1, 2)
    plt.bar(indx, pixel_list)
    plt.xlim(0, 12)
    plt.ylim(0, 5000000)

    plt.tight_layout()
    plt.show()


def img_sum_mask():
    for i in natsort.natsorted(os.listdir(save_path)):
        img = cv2.imread(os.path.join(image_path,i), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(save_path,i), cv2.IMREAD_COLOR)

        sum_img = img*0.6 + mask*0.4

        cv2.imwrite(os.path.join(sum_path, i), sum_img)