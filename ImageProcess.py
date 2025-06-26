import os
import cv2 as cv
import numpy as np
import random
import math
from scipy.spatial import distance

def getDist_P2L(Pointa, Pointb):
    distance = math.sqrt(math.pow(Pointa[0]-Pointb[0], 2) + math.pow(Pointa[1]-Pointb[1], 2))

    return distance

def contour_center(contour):
    M = cv.moments(contour)

    if  M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        point_center = [cX,cY]
    else:
        point_center = [0, 0]

    return point_center

def Find_2max_contours(img):
    # img = cv.bitwise_not(img)
    cv.imwrite('F:/Maize_Root/debing.png',img)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # 找最大面积的边缘
    area = []
    for i in range(len(contours)):
        area.append(cv.contourArea(contours[i]))
    b = sorted(enumerate(area), key=lambda x: x[1], reverse=True)
    max_id = []
    ##计算轮廓中心坐标
    max_id_sel = []
    for i in range(4):
        M = cv.moments(contours[b[i][0]])  # 计算第一条轮廓的各阶矩,字典形式
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        if (center_x<(img.shape[1]/2)):
            max_id_sel.append(b[i][0])
    # print(max_id_sel)

    max_id.append(max_id_sel[0])
    max_id.append(max_id_sel[1])
    # print(max_id)

    for con in range(len(area)):
        if con not in max_id:
            cv.fillPoly(img, [contours[con]], (0))
        else:
            cv.fillPoly(img, [contours[con]], (255))
    # cv.imwrite('debing2.png', img)

    return img, contours[max_id[0]], contours[max_id[1]]

def Find_max_contours(img):
    # img = cv.bitwise_not(img)
    # cv.imwrite('F:/Maize_Root/debing.png',img)
    contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # 找最大面积的边缘
    area = []
    for i in range(len(contours)):
        area.append(cv.contourArea(contours[i]))
    b = sorted(enumerate(area), key=lambda x: x[1], reverse=True)
    max_id = b[0][0]
    ##计算轮廓中心坐标


    for con in range(len(area)):
        if con != max_id:
            cv.fillPoly(img, [contours[con]], (0))
        else:
            cv.fillPoly(img, [contours[con]], (255))


    return img

def Find_max_region(img):
    img2 = FillHole(img)
    contours, hierarchy = cv.findContours(img2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # 找最大面积的边缘
    area = []
    for i in range(len(contours)):
        area.append(cv.contourArea(contours[i]))
    b = sorted(enumerate(area), key=lambda x: x[1], reverse=True)
    max_id = b[0][0]
    ##计算轮廓中心坐标
    for con in range(len(area)):
        if con != max_id:
            cv.fillPoly(img, [contours[con]], (0))


    return img

def show(name,img):
    cv.namedWindow('%s'%name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    width = int((img.shape[1])/2)
    height = int((img.shape[0])/2)
    cv.resizeWindow('%s'%name,width,height)
    cv.imshow('%s'%name, img)

def FillHole(img):
    im_floodfill = img.copy()
    # Mask 用于 floodFill，官方要求长宽+2
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # floodFill函数中的seedPoint必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break
    # 得到im_floodfill
    cv.floodFill(im_floodfill, mask, seedPoint, 255)
    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = img | im_floodfill_inv
    return im_out

def erode_demo(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))#定义结构元素的形状和大小
    dst = cv.erode(image, kernel)#腐蚀操作
    return dst

def open_demo(image):
    # ret, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return binary

def remove_small_objects(img, size):
    #img = cv.bitwise_not(img)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # 找最大面积的边缘
    area = []
    for i in range(len(contours)):
        area.append(cv.contourArea(contours[i]))
    # b = sorted(enumerate(area), key=lambda x: x[1],reverse=True)
    # max_id = b[1][0]
    for con in range(len(area)):
        if area[con] < size:
            cv.fillPoly(img, [contours[con]], (0))
    return img

def get_mask(org, binary_img):
    binary_img = cv.divide(binary_img, 255)
    img_b, img_g, img_r = cv.split(org)
    img_g = img_g * binary_img
    # show('1', img_g)
    img_b = img_b * binary_img
    img_r = img_r * binary_img
    mask_img = cv.merge((img_b, img_g, img_r))
    return mask_img

def close_demo(image):
    ret, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    return binary

def traits_cul(contour):
    traits = []
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h
    width = w
    height = h
    area = cv.contourArea(contour)
    equi_diameter = np.sqrt(4 * area / np.pi)
    perimeter = cv.arcLength(contour, True)
    (x, y), radius = cv.minEnclosingCircle(contour)
    (x, y), (MA, ma), angle = cv.fitEllipse(contour)
    traits.append((width,height,aspect_ratio,area,equi_diameter,perimeter,radius,MA,ma))
    return traits

def point_list(list1):
    list_last = []
    for i in range(len(list1)):
        list_last.append(list1[i][0])
    return list_last

def traits_tools(list2):
    if len(list2)!= 0:
        aver1 = np.average(list2)
        std1 = np.std(list2)
    else:
        aver1 = 0
        std1 = 0

    return aver1, std1

def traits_all(list1):
    number = len(list1)
    width, height, aspect_ratio, area, equi_diameter, perimeter, radius, MA, ma, result1 = [], [], [], [], [], [] ,[] ,[] ,[], []
    for k in range(number):
        area.append(list1[k][0][3])


    area_aver, area_std = traits_tools(area)

    area = []
    area_all = number * area_aver
    for i in range(number):
        if (area_aver - 2 * area_std) <= list1[i][0][3] <= (area_aver + 2 * area_std):
            width.append(list1[i][0][0])
            height.append(list1[i][0][1])
            aspect_ratio.append(list1[i][0][2])
            area.append(list1[i][0][3])
            equi_diameter.append(list1[i][0][4])
            perimeter.append(list1[i][0][5])
            radius.append(list1[i][0][6])
            MA.append(list1[i][0][7])
            ma.append(list1[i][0][8])

    width_aver,width_std = traits_tools(width)
    height_aver, height_std = traits_tools(height)
    aspect_ratio_aver, aspect_ratio_std = traits_tools(aspect_ratio)
    area_aver, area_std = traits_tools(area)
    equi_diameter_aver, equi_diameter_std = traits_tools(equi_diameter)
    perimeter_aver, perimeter_std = traits_tools(perimeter)
    radius_aver, radius_std = traits_tools(radius)
    MA_aver, MA_std = traits_tools(MA)
    ma_aver, ma_std = traits_tools(ma)
    if area_aver != 0:
        number_fitting = area_all / area_aver
    else:
        number_fitting = 0
    result1 = [number,number_fitting, area_all, width_aver, width_std, height_aver, height_std,
            aspect_ratio_aver, aspect_ratio_std,  area_aver,
           area_std,  equi_diameter_aver, equi_diameter_std,  perimeter_aver, perimeter_std,  radius_aver, radius_std,  MA_aver, MA_std, ma_aver, ma_std]

    return result1


def calculate_overlap(contour1, contour2):
    # 计算轮廓的边界矩形
    x1, y1, w1, h1 = cv.boundingRect(contour1)
    x2, y2, w2, h2 = cv.boundingRect(contour2)

    # 计算边界矩形的交集和并集
    intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union = w1 * h1 + w2 * h2 - intersection

    # 计算交并比
    overlap = intersection / union if union > 0 else 0

    return overlap

def remove_connected_regions(image):

    # 找到所有的连通域
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(image, connectivity=8)

    # 确定边界像素的标签
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:,0], labels[:,-1])))

    # 对于所有的边界标签，将它们设置为白色（或黑色，取决于你希望移除的是什么颜色）
    for border_label in border_labels:
        image[labels == border_label] = 0  # 或者 0 如果你想要删除白色连通域

    return image


def add_border(image, border_size=100, border_color=0):

    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    # 在图像周围添加空白边框
    image_with_border = cv.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv.BORDER_CONSTANT, value=border_color)

    return image_with_border


def watershold_components(binary_image):
    binary_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    # 找到连通域
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary_image, connectivity=4)

    output = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), np.uint8)

    # 对每个连通域执行距离变换和分水岭算法
    for i in range(1, num_labels):
        component_mask = np.where(labels == i, 255, 0).astype(np.uint8)


        # 距离变换
        distance_transform = cv.distanceTransform(component_mask, cv.DIST_C, 3)

        # 找到距离变换的最大值，并将它们作为分水岭算法的种子点
        _, sure_fg = cv.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)

        # sure_fg = cv.adaptiveThreshold(distance_transform, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        sure_fg = np.uint8(sure_fg)
        sure_bg = cv.dilate(component_mask, None, iterations=5)
        unknown = cv.subtract(sure_bg, sure_fg)

        # 标记分水岭算法的种子点
        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        # 使用分水岭算法进行分割，只保留分界线，不保留边缘
        component_mask_copy = component_mask.copy()
        component_mask = cv.cvtColor(component_mask, cv.COLOR_GRAY2BGR)
        markers = cv.watershed(component_mask, markers)
        output_now = np.zeros_like(binary_image)
        output_now[markers == -1] = 255

        output_now = cv.bitwise_not(output_now)
        output_now = cv.bitwise_and(output_now, component_mask_copy)

        num_labels_now, labels_now, stats_now, _ = cv.connectedComponentsWithStats(output_now, connectivity=4)

        for j in range(1, num_labels_now):
            # 删除面积小于100的连通域
            if stats_now[j, cv.CC_STAT_AREA] > 100:
                img_j = np.zeros_like(binary_image)
                img_j[labels_now == j] = 255
                img_j = cv.morphologyEx(img_j, cv.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

                img_j = color_connected_components(img_j)
                # 图片相加的时候，如果有重合的区域，将重合的区域变成0
                overlap = np.logical_and(img_j, output)
                output = output + img_j
                output[overlap == 1] = 0

    return output


def color_connected_components(binary_image,):
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary_image, connectivity=4)

    colors = np.zeros((num_labels, 3), dtype=np.uint8)
    colors[1:] = np.random.randint(0, 256, (num_labels-1, 3))  # 0 label is for background

    # 创建彩色结果图像并一次性填充所有像素的颜色
    colored_image = colors[labels]

    return colored_image

def watershold_components_and_cell_classfication(binary_image, image_stele):
    binary_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    # 找到连通域
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary_image, connectivity=4)

    output = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), np.uint8)
    output_center = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), np.uint8)

    # 对每个连通域执行距离变换和分水岭算法
    for i in range(1, num_labels):
        # 计算连通域的中心坐标
        center_x = int(stats[i, cv.CC_STAT_LEFT] + stats[i, cv.CC_STAT_WIDTH] / 2)
        center_y = int(stats[i, cv.CC_STAT_TOP] + stats[i, cv.CC_STAT_HEIGHT] / 2)

        # 判断连通域是否在image_stele中
        if image_stele[center_y, center_x] == 255:
            component_mask = np.where(labels == i, 255, 0).astype(np.uint8)
            # 距离变换
            output = watershold_distance(component_mask, output, color=(50, 50, 'random'))
            # 将中心坐标画在output_center上，用红色
            # output_center = cv.circle(output_center, (center_x, center_y), 1, (0, 0, 255), 5)

        else:

            component_mask = np.where(labels == i, 255, 0).astype(np.uint8)
            output = watershold_distance(component_mask, output, color=('random', 'random', 50))
            # output_center = cv.circle(output_center, (center_x, center_y), 1, (0, 255, 0), 5)

    return output_center


def watershold_distance(component_mask):

    # 距离变换
    distance_transform = cv.distanceTransform(component_mask, cv.DIST_C, 3)

    # 找到距离变换的最大值，并将它们作为分水岭算法的种子点
    _, sure_fg = cv.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    sure_bg = cv.dilate(component_mask, None, iterations=5)
    unknown = cv.subtract(sure_bg, sure_fg)

    # 标记分水岭算法的种子点
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # 使用分水岭算法进行分割，只保留分界线，不保留边缘
    component_mask_copy = component_mask.copy()
    component_mask = cv.cvtColor(component_mask, cv.COLOR_GRAY2BGR)
    markers = cv.watershed(component_mask, markers)

    output_now = np.zeros_like(component_mask_copy)
    output_now[markers == -1] = 255

    output_now = cv.bitwise_not(output_now)
    output_now = cv.bitwise_and(output_now, component_mask_copy)



    # num_labels_now, labels_now, stats_now, _ = cv.connectedComponentsWithStats(output_now, connectivity=4)

    # for j in range(1, num_labels_now):
    #
    #     # 删除面积小于100的连通域
    #     if stats_now[j, cv.CC_STAT_AREA] > 100:
    #         img_j = np.zeros_like(component_mask_copy)
    #         img_j[labels_now == j] = 255
    #
    #         img_j = cv.morphologyEx(img_j, cv.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    #
    #         # 将color中random的位置随机复制
    #         if color[0] == 'random':
    #             color = list(color)
    #             color[0] = np.random.randint(0, 256)
    #             color = tuple(color)
    #         if color[1] == 'random':
    #             color = list(color)
    #             color[1] = np.random.randint(0, 256)
    #             color = tuple(color)
    #         if color[2] == 'random':
    #             color = list(color)
    #             color[2] = np.random.randint(0, 256)
    #             color = tuple(color)
    #
    #         img_j = color_binary_image(img_j, color)
    #         # 图片相加的时候，如果有重合的区域，将重合的区域变成0
    #         overlap = np.logical_and(img_j, output)
    #
    #         output = output + img_j
    #         output[overlap == 1] = 0

    return output_now

# 单个二值图涂色，3个通道值可以定义固定值或者随机值
def color_binary_image(binary_image, color):
    # 初始彩色图像为全0（黑色），大小和二值图一样，但有三个颜色通道
    colored_image = np.zeros((*binary_image.shape, 3), dtype=np.uint8)

    # 在二值图中为真（白色）的地方，设为指定的颜色
    colored_image[binary_image > 0] = color

    return colored_image


def get_max_contour_and_hull(image):
    contour, hi = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_all = []
    area = []
    for num in range(len(contour)):
        contours_all.extend(contour[num])
        area.append(cv.contourArea(contour[num]))

    sorted_stele = sorted(enumerate(area), key=lambda x: x[1], reverse=True)
    contours_max = contour[sorted_stele[0][0]]
    hull = cv.convexHull(np.array(contours_all))

    return contours_max, hull

def min_distance(contour, point):
    return np.min(distance.cdist([point], contour.reshape((-1, 2)), 'euclidean'))

def min_distance_between_two_contour(contour1, contour2):
    return np.min(distance.cdist(contour1.reshape((-1, 2)), contour2.reshape((-1, 2)), 'euclidean'))

def img_resize(img, size):
    img_new = np.zeros((1024, 1024, 3), np.uint8)
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if img.shape[0] > img.shape[1]:
        img = cv.resize(img, (int(size * img.shape[1] / img.shape[0]), size), interpolation=cv.INTER_AREA)
        img_new[:, int((size - img.shape[1]) / 2):int((size - img.shape[1]) / 2) + img.shape[1], :] = img
    else:
        img = cv.resize(img, (size, int(size * img.shape[0] / img.shape[1])), interpolation=cv.INTER_AREA)
        img_new[int((size - img.shape[0]) / 2):int((size - img.shape[0]) / 2) + img.shape[0], :, :] = img
    return img_new

def Check_stele_cell(img, annotation, img_cortex_all, contour_section):
    num, labels, stats, _ = cv.connectedComponentsWithStats(
        img, connectivity=4)

    # 所有连通域最外圈轮廓
    contours, _ = cv.findContours(labels.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # contours合成一个
    contours = np.vstack(contours)

    img_ellipse = np.zeros_like(img)

    ellipse = cv.fitEllipse(contours)
    (xc, yc), (major, minor), angle = ellipse
    # 膨胀椭圆的长轴和短轴（可调倍率），原来是1.2
    scale = 1.2
    major *= scale
    minor *= scale

    # 构建放大后的椭圆
    scaled_ellipse = ((xc, yc), (major, minor), angle)

    cv.ellipse(img_ellipse, scaled_ellipse, 255, 2)
    img_ellipse_fill = FillHole(img_ellipse)

    contours_ellipse, contour_hull_ellipse = get_max_contour_and_hull(img_ellipse_fill)

    # 判断img的所有连通域是否在img_ellipse内
    for i in range(1, num):
        center_x_component_pro = int(
            stats[i, cv.CC_STAT_LEFT] + stats[i, cv.CC_STAT_WIDTH] / 2)
        center_y_component_pro = int(
            stats[i, cv.CC_STAT_TOP] + stats[i, cv.CC_STAT_HEIGHT] / 2)

        distance_stele = cv.pointPolygonTest(contours_ellipse, (center_x_component_pro, center_y_component_pro), True)

        # 取出连通域
        mask = np.zeros_like(img)
        mask[labels == i] = 255
        mask = cv.morphologyEx(mask, cv.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
        # mask是否在img_ellipse内
        mask_ellipse = cv.bitwise_and(mask, img_ellipse_fill)

        contours_i, contour_hull = get_max_contour_and_hull(mask)

        distance_stele_min_contour = min_distance_between_two_contour(contours_ellipse, contours_i)


        if np.count_nonzero(mask_ellipse) == 0 and stats[i, cv.CC_STAT_AREA] > 100:
            img_cortex_all[labels == i] = 255
            annotation['annotations'].append({'category_id': '0', 'category_name': 'cortex cell',
                                                        'contours': contours_i.tolist(),
                                                        'area': stats[i, cv.CC_STAT_AREA],
                                                        'center_distance_stele_min': distance_stele,
                                                        'center_distance_section_min': 100000,
                                                        'contour_distance_stele_min': distance_stele_min_contour,
                                                        'contour_distance_section_min': 100000})
        else:
            annotation['annotations'].append({'category_id': '0', 'category_name': 'stele cell',
                                                        'contours': contours_i.tolist(),
                                                        'area': stats[i, cv.CC_STAT_AREA],
                                                        'center_distance_stele_min': distance_stele,
                                                        'center_distance_section_min': None,
                                                        'contour_distance_stele_min': distance_stele_min_contour,
                                                        'contour_distance_section_min': None})


    # 判定cortex部分
    (xc, yc), (major, minor), angle = ellipse
    scale = 1.25
    major *= scale
    minor *= scale

    # 构建放大后的椭圆
    scaled_ellipse = ((xc, yc), (major, minor), angle)

    cv.ellipse(img_ellipse, scaled_ellipse, 255, 2)
    img_ellipse_fill = FillHole(img_ellipse)

    contours_ellipse, contour_hull_ellipse = get_max_contour_and_hull(img_ellipse_fill)
    num_cortex, labels_cortex, stats_cortex, _ = cv.connectedComponentsWithStats(
        img_cortex_all, connectivity=4)

    img_cortex_all_close = cv.morphologyEx(img_cortex_all, cv.MORPH_CLOSE, np.ones((30, 30), np.uint8))
    contour_in = Find_max_in_contour(img_cortex_all_close)


    for j in range(1, num_cortex):
        center_x_component_pro = int(
            stats_cortex[j, cv.CC_STAT_LEFT] + stats_cortex[j, cv.CC_STAT_WIDTH] / 2)
        center_y_component_pro = int(
            stats_cortex[j, cv.CC_STAT_TOP] + stats_cortex[j, cv.CC_STAT_HEIGHT] / 2)

        distance_stele = cv.pointPolygonTest(contour_in, (center_x_component_pro, center_y_component_pro),
                                             True)

        # 计算连通域中心与contour_section的距离
        distance_section = cv.pointPolygonTest(contour_section, (center_x_component_pro, center_y_component_pro),
                                               True)
        # 取出连通域
        mask = np.zeros_like(img)
        mask[labels_cortex == j] = 255
        mask = cv.morphologyEx(mask, cv.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

        contours_j, contour_hull = get_max_contour_and_hull(mask)

        distance_stele_min_contour = min_distance_between_two_contour(contour_in, contours_j)
        distance_section_min_contour = min_distance_between_two_contour(contour_section, contours_j)

        if  stats_cortex[j, cv.CC_STAT_AREA] > 100:
            annotation['annotations'].append({'category_id': '0', 'category_name': 'cortex cell',
                                              'contours': contours_j.tolist(),
                                              'area': stats_cortex[j, cv.CC_STAT_AREA],
                                              'center_distance_stele_min': distance_stele,
                                              'center_distance_section_min': distance_section,
                                              'contour_distance_stele_min': distance_stele_min_contour,
                                              'contour_distance_section_min': distance_section_min_contour})


    return img_ellipse, annotation, img_cortex_all

def Find_max_in_contour(img):

    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # 4. 找到面积最大的内轮廓
    max_area = 0
    max_contour = None

    # 注意 hierarchy[0][i][3] == -1 是外轮廓，非 -1 则为内轮廓
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:  # 只保留内轮廓
            area = cv.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

    return max_contour


def calculate_mask_overlap_ratio(component_mask_test, mask):
    """
    计算 component_mask_test 在 mask 内的比例

    参数:
        component_mask_test: 待测试的二值掩码（numpy数组）
        mask: 参考的二值掩码（numpy数组）

    返回:
        overlap_ratio: component_mask_test 中落在 mask 内的像素比例 (0~1)
    """
    # 确保两个掩码形状相同
    assert component_mask_test.shape == mask.shape

    # 将掩码转换为布尔类型
    component_mask_test = component_mask_test.astype(bool)
    mask = mask.astype(bool)

    # 计算 component_mask_test 与 mask 的交集
    intersection = np.logical_and(component_mask_test, mask)

    # 计算 component_mask_test 的总像素数（避免除以0）
    total_pixels = np.sum(component_mask_test)
    if total_pixels == 0:
        return 0.0  # 如果 component_mask_test 为空，则比例为0

    # 计算重叠比例
    overlap_ratio = np.sum(intersection) / total_pixels

    return overlap_ratio
