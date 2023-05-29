import torch
import numpy as np
from torch.nn import functional as F

def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
    batch, _, ori_height, ori_width = image.size()
    assert batch == 1, "only supporting batchsize 1."
    image = image.numpy()[0].transpose((1,2,0)).copy()
    stride_h = np.int(self.crop_size[0] * 1.0)
    stride_w = np.int(self.crop_size[1] * 1.0)
    final_pred = torch.zeros([1, self.num_classes,
                                ori_height,ori_width]).cuda()
    for scale in scales:
        new_img = self.multi_scale_aug(image=image,
                                       rand_scale=scale,
                                       rand_crop=False)
        height, width = new_img.shape[:-1]

        if scale <= 1.0:
            new_img = new_img.transpose((2, 0, 1))
            new_img = np.expand_dims(new_img, axis=0)
            new_img = torch.from_numpy(new_img)
            preds = self.inference(config, model, new_img, flip)
            preds = preds[:, :, 0:height, 0:width]
        else:
            new_h, new_w = new_img.shape[:-1]
            rows = np.int(np.ceil(1.0 * (new_h - 
                            self.crop_size[0]) / stride_h)) + 1
            cols = np.int(np.ceil(1.0 * (new_w - 
                            self.crop_size[1]) / stride_w)) + 1
            preds = torch.zeros([1, self.num_classes,
                                       new_h,new_w]).cuda()
            count = torch.zeros([1,1, new_h, new_w]).cuda()

            for r in range(rows):
                for c in range(cols):
                    h0 = r * stride_h
                    w0 = c * stride_w
                    h1 = min(h0 + self.crop_size[0], new_h)
                    w1 = min(w0 + self.crop_size[1], new_w)
                    h0 = max(int(h1 - self.crop_size[0]), 0)
                    w0 = max(int(w1 - self.crop_size[1]), 0)
                    crop_img = new_img[h0:h1, w0:w1, :]
                    crop_img = crop_img.transpose((2, 0, 1))
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_img = torch.from_numpy(crop_img)
                    pred = self.inference(config, model, crop_img, flip)
                    preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                    count[:,:,h0:h1,w0:w1] += 1
            preds = preds / count
            preds = preds[:,:,:height,:width]

        preds = F.interpolate(
            preds, (ori_height, ori_width), 
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )            
        final_pred += preds
    return final_pred
