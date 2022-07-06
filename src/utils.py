import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tff

def mpp_extractor(file_object):
    for string in file_object:
        find_flag = string.find('X Dimension')
        if find_flag != -1:
            start = string.find('[um],') + len('[um],')
            end = string.find('[um/pixel]')
            mpp = string[start:end]
            return float(mpp)
        
        
def get_cnt_from_df(df, x_name='x', y_name='y'):
    xs, ys = df[x_name], df[y_name]
    cnt = [[[x, y]] for x, y in zip(xs, ys)]
    cnt.append(cnt[0])
    cnt = np.array(cnt, dtype = int)
    return cnt

def draw_scale(img, mpp, length=5):
    img = img.copy()
    start_point = img.shape[1] * 0.05, img.shape[0] * 0.95
    start_point = list(map(int, start_point))
    end_point = [start_point[0] + length / mpp, start_point[1]]
    end_point = list(map(int, end_point))
    end_point[0] = min((end_point[0], img.shape[1]))
    thickness = 2 * max(img.shape) / 512 
    thickness = int(thickness)
    img = cv2.line(img, start_point, end_point, (255, 255, 0), thickness)
    
    start_point[1] -= int(img.shape[0] * 0.02)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = start_point
    fontScale              = 0.9
    fontColor              = (255,255,255)
    thickness              = thickness
    lineType               = 2

    cv2.putText(img,
                f'{length}um', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType
                )

    return img

def draw_annotation(img, cnts_d):
    img = img.copy()
    img -= img.min()
    img = img.astype(float)
    img /= (img.max())
    img = (255. * img).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    tube_cnts = [d['tube'] for d in cnts_d['tube_rois']] 
    bg_tube_cnts = [d['bg_tube'] for d in cnts_d['tube_rois']]
    thickness = 2 * max(img.shape) / 512
    thickness = int(thickness)
    cv2.polylines(img, cnts_d['membrane_rois']['membranes_cnts'], color= (0, 0, 255), isClosed=True, thickness=thickness)
    cv2.polylines(img, cnts_d['membrane_rois']['bg_cnts'], color= (0, 0, 255), isClosed=True, thickness=thickness)
    cv2.polylines(img, tube_cnts + bg_tube_cnts, color= (0, 255, 0), isClosed=True, thickness=thickness)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def visualize_ds(df, ncols, figsize):
    nrows = len(df) / ncols
    nrows = int(np.ceil(nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for ax, (_, row) in zip(axes.flatten(), df.iterrows()):
        img = tff.imread(row['image_path'])
        img = draw_annotation(img, row['cnts_d'])
        img = draw_scale(img, row['mpp'])
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('./dataset.png')