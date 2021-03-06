import numpy as np
import cfg


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


def creat_roiheatmap_ellipse(det_size_map):
    sigma_x = ((det_size_map[1] - 1) * 0.5 - 1) * 0.3 + 0.8
    s_x = 2 * (sigma_x ** 2)
    sigma_y = ((det_size_map[0] - 1) * 0.5 - 1) * 0.3 + 0.8
    s_y = 2 * (sigma_y ** 2)
    h, w = [(hw - 1.) / 2. for hw in det_size_map]
    y, x = np.ogrid[-h:h + 1, -w:w + 1]
    heatmap = np.exp(-x**2 / s_x - y**2 / s_y)
    return heatmap

def creat_roiheatmap_circle(det_size_map):
    min_size=min(det_size_map)
    sigma = ((min_size - 1) * 0.5 - 1) * 0.3 + 0.8
    s_ = 2 * (sigma ** 2)
    h, w = [(hw - 1.) / 2. for hw in det_size_map]
    y, x = np.ogrid[-h:h + 1, -w:w + 1]
    heatmap = np.exp(-x**2 / s_ - y**2 / s_)
    return heatmap

def CreatGroundTruth(label_batch):

    batch = len(label_batch)
    cls_gt_batch = np.zeros(shape=[batch, cfg.featuremap_h, cfg.featuremap_w, cfg.num_classes], dtype=np.float32)
    size_gt_batch = np.zeros(shape=[batch, cfg.featuremap_h, cfg.featuremap_w, 2], dtype=np.float32)

    for x in range(batch):

        for n in range(len(label_batch[x]) // 5):
            class_id, x_min, y_min, x_max, y_max = int(label_batch[x][n * 5]), float(label_batch[x][n * 5 + 1]), float(
                label_batch[x][n * 5 + 2]), float(label_batch[x][n * 5 + 3]), float(label_batch[x][n * 5 + 4])

            x_min_map = int(np.floor(x_min / cfg.down_ratio))
            y_min_map = int(np.floor(y_min / cfg.down_ratio))
            x_max_map = int(np.floor(x_max / cfg.down_ratio))
            y_max_map = int(np.floor(y_max / cfg.down_ratio))
            size_map_int = (y_max_map - y_min_map, x_max_map - x_min_map)

            size_ori = [x_max - x_min, y_max - y_min]  # w*h
            size_map_float = [size_ori[0] / cfg.down_ratio, size_ori[1] / cfg.down_ratio]

            # Official implementation ? not sure,It is currently certain that the heat map exceeds the target bbox and the loss will not converge.
            # radius = gaussian_radius(size_map_float)
            # radius = radius if radius != 0.0 else 2.5
            # radius=sigma ？ Need to be further resolved
            # draw_msra_gaussian(cls_gt_batch[x, :, :, class_id], center_map_int, radius)

            # My modified implementation
            center_ori = [x_min + size_ori[0] / 2.0, y_min + size_ori[1] / 2.0]  # x,y
            center_map = [center_ori[0] / cfg.down_ratio, center_ori[1] / cfg.down_ratio]
            center_map_int = [int(center_map[0]), int(center_map[1])]
           
            #you can choose circle or ellipse
            #heatmap_roi = creat_roiheatmap_circle(size_map_int)
            heatmap_roi = creat_roiheatmap_ellipse(size_map_int)
            cls_gt_batch[x, y_min_map:y_max_map, x_min_map:x_max_map, class_id] = np.maximum(
                cls_gt_batch[x, y_min_map:y_max_map, x_min_map:x_max_map, class_id], heatmap_roi)

            row1 = center_map_int[1] - 1
            row2 = center_map_int[1] + 2
            col1 = center_map_int[0] - 1
            col2 = center_map_int[0] + 2
            size_gt_batch[x, row1:row2, col1:col2, 0] = size_map_float[0]
            size_gt_batch[x, row1:row2, col1:col2, 1] = size_map_float[1]

    return cls_gt_batch, size_gt_batch
