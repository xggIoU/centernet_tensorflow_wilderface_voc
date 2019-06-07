import numpy as np
import cfg
import cv2


def gaussian_radius(det_size, min_overlap=0.7):
    width,height  = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0])
    mu_y = int(center[1])
    h, w = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def creat_roiheatmap(centern_roi, det_size_map):
    c_x, c_y = centern_roi
    sigma_x = ((det_size_map[1] - 1) * 0.5 - 1) * 0.3 + 0.8
    s_x = 2 * (sigma_x ** 2)
    sigma_y = ((det_size_map[0] - 1) * 0.5 - 1) * 0.3 + 0.8
    s_y = 2 * (sigma_y ** 2)
    X1 = np.arange(det_size_map[1])
    Y1 = np.arange(det_size_map[0])
    [X, Y] = np.meshgrid(X1, Y1)
    heatmap = np.exp(-(X - c_x) ** 2 / s_x - (Y - c_y) ** 2 / s_y)
    return heatmap


def CreatGroundTruth(label_batch):
    batch = len(label_batch)
    center_gt_batch = np.zeros(shape=[batch, cfg.featuremap_h, cfg.featuremap_w, cfg.num_classes], dtype=np.float32)
    offset_gt_batch = np.zeros(shape=[batch, cfg.featuremap_h, cfg.featuremap_w, 2], dtype=np.float32)
    size_gt_batch = np.zeros(shape=[batch, cfg.featuremap_h, cfg.featuremap_w, 2], dtype=np.float32)
    mask_gt_batch = np.zeros(shape=[batch, cfg.featuremap_h, cfg.featuremap_w], dtype=np.float32)

    for x in range(batch):
        for n in range(len(label_batch[x]) // 5):
            class_id, x_min, y_min, x_max, y_max = int(label_batch[x][n * 5]), float(label_batch[x][n * 5 + 1]), float(
                label_batch[x][n * 5 + 2]), float(label_batch[x][n * 5 + 3]), float(label_batch[x][n * 5 + 4])

            x_min_map = np.floor(x_min / cfg.down_ratio)
            y_min_map = np.floor(y_min / cfg.down_ratio)
            x_max_map = np.floor(x_max / cfg.down_ratio)
            y_max_map = np.floor(y_max / cfg.down_ratio)
            size_map_int = (y_max_map - y_min_map, x_max_map - x_min_map)

            size_ori=[x_max-x_min,y_max-y_min]#w*h
            size_map_float=[size_ori[0]/cfg.down_ratio,size_ori[1]/cfg.down_ratio]
            center_ori=[x_min+size_ori[0]/2.0,y_min+size_ori[1]/2.0]#x,y
            center_map=[center_ori[0]/cfg.down_ratio,center_ori[1]/cfg.down_ratio]
            center_map_int = [int(center_map[0]),int(center_map[1])]

            # center_map_obj = [center_map_int[0]-x_min_map,center_map_int[1]-y_min_map]
            # heatmap_roi = creat_roiheatmap(center_map_obj, size_map_int)
            # center_gt_batch[x, int(y_min_map):int(y_max_map), int(x_min_map):int(x_max_map), class_id] = np.maximum(
            #     center_gt_batch[x, int(y_min_map):int(y_max_map), int(x_min_map):int(x_max_map), class_id], heatmap_roi)
            radius = gaussian_radius(size_map_float)
            radius = radius if radius != 0.0 else 2.5
            draw_msra_gaussian(center_gt_batch[x, :, :, class_id], center_map_int, radius)

            offset=np.asarray(center_map)-np.asarray(center_map_int)

            row1 = int(center_map_int[1]) - 1
            row2 = int(center_map_int[1]) + 2
            col1 = int(center_map_int[0]) - 1
            col2 = int(center_map_int[0]) + 2
            offset_gt_batch[x, row1:row2, col1:col2, 0] = offset[0]
            offset_gt_batch[x, row1:row2, col1:col2, 1] = offset[1]
            size_gt_batch[x, row1:row2, col1:col2, 0] = size_map_float[0]
            size_gt_batch[x, row1:row2, col1:col2, 1] = size_map_float[1]
            mask_gt_batch[x, row1:row2, col1:col2] = 1.0
            # cv2.waitKey()
            # print(np.sum(mask_gt_batch))
    # center_gt_batch=(center_gt_batch>0.6).astype(np.float32)
    # for b in range(cfg.num_classes):
    #     print(b)
    #     cv2.imshow('hm',center_gt_batch[0,:,:,b])
    # #     cv2.imshow('hm.0',len(center_gt_batch[0, :, :, b][center_gt_batch[0, :, :, b] > 0.0]))
    # #     print(np.sum((center_gt_batch[0, :, :, b] > 0.0).astype(np.float32)))
    #     cv2.waitKey()

    return center_gt_batch, offset_gt_batch, size_gt_batch, mask_gt_batch


if __name__ == '__main__':
    print(gaussian_radius((16, 16)))