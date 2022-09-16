import os.path

import cv2
import numpy as np

CLASSES = ['background', 'bottle', 'can', 'chain',
           'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
           'standing-bottle', 'tire', 'valve', 'wall']

# opencv image BGR
color_map = {
    '0': [0, 0, 0], # black
    '1': [0, 0, 255], # red - bottle
    '2': [0, 125, 255], # orange - can
    '3': [0, 255, 255], # yellow - chain
    '4': [0, 255, 125], # spring green - drink-carton
    '5': [0, 255, 0], # green - hook
    '6': [255, 255, 0], # cyan - propeller
    '7': [255, 125, 0], # ocean - shampoo-bottle
    '8': [255, 0, 0], # blue - standing-bottle
    '9': [255, 0, 125], # violet - tire
    '10': [255, 0, 255], # magenta - valve
    '11': [125, 0, 255], # raspberry - wall
}

def pred_colorization(imgs, gt, mask, epoch, val_idx):
    imgs = imgs.squeeze() * 255
    imgs.astype(np.uint8)

    imgs = cv2.cvtColor(imgs, cv2.COLOR_GRAY2BGR)

    gt_color = np.zeros((480 * 320, 3)).astype(np.uint8)
    mask_color = np.zeros((480 * 320, 3)).astype(np.uint8)

    obj_list = [99999999]*12

    for i in range(0, len(color_map)):
        idx_gt = np.where(gt==int(i))
        idx_mask = np.where(mask==int(i))

        if int(i) != 0:
            if len(idx_gt[0])!=0:
                obj_list[i] = idx_gt[0][0]

        gt_color[idx_gt,:] = color_map['{}'.format(i)]
        mask_color[idx_mask,:] = color_map['{}'.format(i)]

    gt_color = gt_color.reshape((480,320,3))
    mask_color = mask_color.reshape((480,320,3))

    img_pred = imgs * 0.7 + mask_color * 0.3

    idx_gt_w = np.where(gt>0)
    idx_mask_w = np.where(mask>0)

    img_overlap = np.zeros((480 * 320, 3)).astype(np.uint8)
    img_overlap[idx_gt_w, 1] = 255
    img_overlap[idx_mask_w, 2] = 255
    img_overlap = img_overlap.reshape((480,320,3))

    img_hstack = np.hstack((gt_color, mask_color, img_overlap, imgs, img_pred))

    cv2.putText(img_hstack, 'GT', (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img_hstack, 'Pred', (330,30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img_hstack, 'GT+Pred', (650,30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img_hstack, 'img', (970,30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img_hstack, 'img+pred', (1290,30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)

    for i, obj in enumerate(CLASSES):
        if obj_list[i]!=99999999:
            h, w = divmod(int(obj_list[i]), 320)
            cv2.putText(img_hstack, obj, (w, h), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)

    cv2.imwrite("./vis/val/{}/{}.png".format(epoch, val_idx), img_hstack)



