import os

import cv2

from ImageProcess import *
from scipy.spatial import distance
import matplotlib.pyplot as plt
from tqdm import tqdm

class  RootTraitCal:
    def __init__(self, img_name, img_stele, img_cortex, DS_pericycle, DS2_pericycle, Area_pericycle, DS1_endodermis,
                                   DS2_endodermis, Area_endodermis, DS1_exodermis, DS2_exodermis, Area_exodermis):
        self.img_name = img_name
        self.img_stele = img_stele
        self.img_cortex = img_cortex
        self.img_stele_pro = None
        self.img_cortex_pro = None
        self.DS_pericycle = DS_pericycle
        self.DS2_pericycle = DS2_pericycle
        self.Area_pericycle = Area_pericycle
        self.DS1_endodermis = DS1_endodermis
        self.DS2_endodermis = DS2_endodermis
        self.Area_endodermis = Area_endodermis
        self.DS1_exodermis = DS1_exodermis
        self.DS2_exodermis = DS2_exodermis
        self.Area_exodermis = Area_exodermis
        self.cell_area = {'area': []}
        # self.img_last = np.zeros((img_stele.shape[0], img_stele.shape[1], 3), dtype=np.uint8)
        self.cell_annotation = {'image': [], 'annotations': []}
        filename = os.path.splitext(self.img_name)
        self.cell_annotation['image'].append({'image_name': filename[0]})

    def preprocess(self):
        index, self.img_stele_pro = cv.threshold(self.img_stele, 1, 255, cv.THRESH_BINARY)
        self.img_stele_pro = add_border(self.img_stele_pro)
        index, self.img_cortex_pro = cv.threshold(self.img_cortex, 1, 255, cv.THRESH_BINARY)
        self.img_cortex_pro = add_border(self.img_cortex_pro)

        img_section_pro_origin = cv.add(self.img_stele_pro, self.img_cortex_pro)
        # image close to connect some area
        img_section_pro = cv.medianBlur(img_section_pro_origin, 3)
        kernel_close = np.ones((40, 40), dtype=np.uint8)
        img_section_pro_close = cv.morphologyEx(img_section_pro, cv.MORPH_CLOSE, kernel_close)
        img_section_pro_close = FillHole(img_section_pro_close)
        kernel_open = np.ones((30, 30), dtype=np.uint8)
        img_section_pro_close = cv2.morphologyEx(img_section_pro_close, cv.MORPH_OPEN, kernel_open)

        # get max area
        img_section_pro_close = Find_max_region(img_section_pro_close)

        # mask to origin image
        img_section_pro_close = cv.bitwise_and(img_section_pro_close, img_section_pro_origin)

        # close the image and find the contour, then add to the image
        img_section_pro_close2 = FillHole(img_section_pro_close)
        img_section_pro_close2 = remove_small_objects(img_section_pro_close2, 100)
        kernel_close = np.ones((60, 60), dtype=np.uint8)
        img_section_pro_close2 = cv.morphologyEx(img_section_pro_close2, cv.MORPH_CLOSE, kernel_close)
        img_section_pro_close2 = Find_max_region(img_section_pro_close2)
        img_section_pro_close2 = FillHole(img_section_pro_close2)
        laplace_cortex = cv.Laplacian(img_section_pro_close2, cv.CV_8U, ksize=5)

        # mask to origin image
        img_section_pro_mask = img_section_pro_close2.copy()
        img_section_pro_mask = cv.divide(img_section_pro_mask, 255)
        self.img_section_pro_mask = img_section_pro_mask.copy()

        # the region of the entire section
        img_section_pro_close = cv.add(img_section_pro_close, laplace_cortex)
        self.img_section_pro = cv.multiply(img_section_pro_close, img_section_pro_mask)

        # the region of the stele
        self.img_stele_pro = cv.multiply(self.img_stele_pro, img_section_pro_mask)
        kernel_close2 = np.ones((30, 30), dtype=np.uint8)
        self.img_stele_pro = cv.morphologyEx(self.img_stele_pro, cv.MORPH_CLOSE, kernel_close2)
        img_stele_mask = FillHole(self.img_stele_pro)
        img_stele_mask = cv.morphologyEx(img_stele_mask, cv.MORPH_OPEN, kernel_close2)
        self.img_stele_mask = img_stele_mask.copy()

        self.img_last = np.zeros((img_section_pro.shape[0], img_section_pro.shape[1], 3), np.uint8)

        # self.img_stele_pro转为三通道，换个颜色
        # self.img_stele_pro = cv.cvtColor(self.img_stele_pro, cv.COLOR_GRAY2BGR)
        # mask = (self.img_stele_pro != [0, 0, 0]).any(axis=-1)
        # # Assign the new value to the pixels where the mask is True
        # self.img_stele_pro[mask] = [50, 50, 50]
        #
        # self.img_last = self.img_last + self.img_stele_pro
        #
        # cv.imwrite('show_img/1_' + self.img_name, self.img_last)

        # new
        # self.img_stele_pro = cv.multiply(self.img_stele_pro, img_section_pro_mask)
        # self.img_stele_pro = FillHole(self.img_stele_pro)
        # kernel_close2  = np.ones((300, 300), dtype=np.uint8)
        # self.img_stele_pro = cv.morphologyEx(self.img_stele_pro, cv.MORPH_CLOSE, kernel_close2)
        # img_stele_mask = FillHole(self.img_stele_pro)
        # img_stele_mask = cv.morphologyEx(img_stele_mask, cv.MORPH_OPEN, kernel_close2)
        # self.img_stele_mask = img_stele_mask.copy()




        # the outer contour and hull of the stele and the entire section
        self.contour_stele, self.hull_stele = get_max_contour_and_hull(img_stele_mask)
        self.contour_section, self.hull_section = get_max_contour_and_hull(img_section_pro_mask)


    def cell_detection(self):
        # 图像反色并标记连通域

        img_section_pro = cv.bitwise_not(self.img_section_pro)
        img_section_pro = cv.multiply(img_section_pro, self.img_section_pro_mask)

        # 开操作分开一部分细胞
        img_section_pro = cv.morphologyEx(img_section_pro, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

        # 找到连通域
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(img_section_pro, connectivity=4)

        self.img_stele_all = np.zeros((img_section_pro.shape[0], img_section_pro.shape[1]), np.uint8)
        self.img_cortex_all = np.zeros((img_section_pro.shape[0], img_section_pro.shape[1]), np.uint8)

        # 对每个连通域执行距离变换和分水岭算法
        cell_area = []
        for i in tqdm(range(1, num_labels)):

            x_comp, y_comp, width_comp, height_comp, area_comp = stats[i]
            # 计算连通域的中心坐标
            center_x = int(stats[i, cv.CC_STAT_LEFT] + stats[i, cv.CC_STAT_WIDTH] / 2)
            center_y = int(stats[i, cv.CC_STAT_TOP] + stats[i, cv.CC_STAT_HEIGHT] / 2)

            # 判断连通域是否在image_stele中
            component_mask_test = np.where(labels == i, 255, 0).astype(np.uint8)
            iou = calculate_mask_overlap_ratio(component_mask_test, self.img_stele_mask)
            # print(iou)
            if self.img_stele_mask[center_y, center_x] == 255 and iou > 0.4:

                # component_mask = np.where(labels == i, 255, 0).astype(np.uint8)
                # # 距离变换
                # component_pro = watershold_distance(component_mask)
                # # 计算每个连通域的信息并进行分类
                # num_labels_component_pro, labels_component_pro, stats_component_pro, _ = cv.connectedComponentsWithStats(component_pro, connectivity=4)

                binary_i = (labels[y_comp - 10:y_comp + 10 + height_comp,
                            x_comp - 10:x_comp + 10 + width_comp] == i).astype(np.uint8) * 255

                # 距离变换
                component_pro = watershold_distance(binary_i)
                # 计算每个连通域的信息并进行分类
                num_labels_component_pro, labels_component_pro, stats_component_pro, _ = cv.connectedComponentsWithStats(
                    component_pro, connectivity=4)

                for j in range(1, num_labels_component_pro):
                    if stats_component_pro[j, cv.CC_STAT_AREA] > 20:
                        cell_area.append(stats_component_pro[j, cv.CC_STAT_AREA])
                        self.stele_area = cell_area
                        # 在self.img_stele_last上绘制连通域
                        y0 = y_comp - 10
                        x0 = x_comp - 10
                        mask = (labels_component_pro == j).astype(np.uint8)
                        self.img_stele_all[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]][mask == 1] = 255

            else:
                binary_i = (labels[y_comp - 10:y_comp + 10 + height_comp,
                            x_comp - 10:x_comp + 10 + width_comp] == i).astype(np.uint8) * 255

                # 距离变换
                component_pro = watershold_distance(binary_i)
                # 计算每个连通域的信息并进行分类
                num_labels_component_pro, labels_component_pro, stats_component_pro, _ = cv.connectedComponentsWithStats(
                    component_pro, connectivity=4)

                # component_mask = np.where(labels == i, 255, 0).astype(np.uint8)
                # # 距离变换
                # component_pro = watershold_distance(component_mask)
                # # 计算每个连通域的信息并进行分类
                # num_labels_component_pro, labels_component_pro, stats_component_pro, _ = cv.connectedComponentsWithStats(
                #     component_pro, connectivity=4)

                for k in range(1, num_labels_component_pro):
                    if stats_component_pro[k, cv.CC_STAT_AREA] > 150:
                        y0 = y_comp - 10
                        x0 = x_comp - 10
                        mask = (labels_component_pro == k).astype(np.uint8)
                        self.img_cortex_all[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]][mask == 1] = 255

    def cell_class(self):
        index = 0

        # cv2.imwrite('ceshi/cortex' + self.img_name, self.img_stele_all)
        # cv2.imwrite('ceshi/stele' +  self.img_name,  self.img_cortex_all)

        self.img_stele_ellipse, self.cell_annotation, self.img_cortex_all = Check_stele_cell(self.img_stele_all, self.cell_annotation, self.img_cortex_all, self.contour_section)
        # cv2.imwrite('data/output_cortex/' + name, self.img_cortex_all)

        for cell in self.cell_annotation['annotations']:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            if cell['category_name'] == 'stele cell':
                # 判断中柱鞘
                if cell['contour_distance_stele_min'] < self.DS_pericycle and cell['center_distance_stele_min'] < self.DS2_pericycle and cell['area'] < self.Area_pericycle:
                    self.cell_annotation['annotations'][index]['category_id'] = '3'
                    self.cell_annotation['annotations'][index]['category_name'] = 'pericycle'
                    self.cell_area['area'].append({'pericycle': cell['area']})
                    self.img_last = cv.fillPoly(self.img_last, [np.array(cell['contours'])], (255,255,255))
                elif cell['area'] >  np.average(self.stele_area) + 4 * np.std(self.stele_area):
                    self.cell_annotation['annotations'][index]['category_id'] = '1'
                    self.cell_annotation['annotations'][index]['category_name'] = 'metaxylem'
                    self.cell_area['area'].append({'metaxylem': cell['area']})
                    self.img_last = cv.fillPoly(self.img_last, [np.array(cell['contours'])], (226, 43, 138))
                else:
                    self.cell_annotation['annotations'][index]['category_id'] = '2'
                    self.cell_annotation['annotations'][index]['category_name'] = 'stele'
                    self.cell_area['area'].append({'stele': cell['area']})
                    self.img_last = cv.fillPoly(self.img_last, [np.array(cell['contours'])], (50, 50, r))


            elif cell['category_name'] == 'cortex cell':
                # 内皮层 值越大预测的就越广
                if  cell['center_distance_stele_min'] < self.DS2_endodermis and cell['contour_distance_stele_min'] < self.DS2_endodermis and cell['area'] < self.Area_endodermis:
                    self.cell_annotation['annotations'][index]['category_id'] = '4'
                    self.cell_annotation['annotations'][index]['category_name'] = 'endodermis'
                    self.cell_area['area'].append({'endodermis': cell['area']})
                    self.img_last = cv.fillPoly(self.img_last, [np.array(cell['contours'])], (192, 192, 192))
                # 外皮曾 一样
                elif cell['center_distance_section_min'] < self.DS2_exodermis and cell['contour_distance_section_min'] < self.DS1_exodermis and cell['area'] < self.Area_exodermis:
                    self.cell_annotation['annotations'][index]['category_id'] = '6'
                    self.cell_annotation['annotations'][index]['category_name'] = 'epidermis'
                    self.cell_area['area'].append({'epidermis': cell['area']})
                    self.img_last = cv.fillPoly(self.img_last, [np.array(cell['contours'])], (0, 208, 244))
                else:
                    self.cell_annotation['annotations'][index]['category_id'] = '5'
                    self.cell_annotation['annotations'][index]['category_name'] = 'cortex'
                    self.cell_area['area'].append({'cortex': cell['area']})
                    self.img_last = cv.fillPoly(self.img_last, [np.array(cell['contours'])], (b, g, 50))

            index += 1

    def trait_cal(self):
        category_name = [row['category_name'] for row in self.cell_annotation['annotations']]
        category_name = set(category_name)
        category_name = sorted(category_name)
        self.category_name = list(category_name)
        self.category_num = len(category_name)

        # Create a new list of part names
        for i in range(len(category_name)):
            globals()[self.category_name[i]] = []

        for cell in self.cell_annotation['annotations']:
            if cell['category_name'] in category_name:
                index = self.category_name.index(cell['category_name'])
                globals()[self.category_name[index]].append(traits_cul(np.array(cell['contours'])))

        traits_name = []
        self.traits_name_all = ['number', 'number_fitting', 'area_all', 'width_aver', 'width_std', 'height_aver', 'height_std', 'aspect_ratio_aver', 'aspect_ratio_std',
                           'area_aver', 'area_std', 'equi_diameter_aver', 'equi_diameter_std', 'perimeter_aver', 'perimeter_std', 'radius_aver', 'radius_std',
                           'MA_aver',	'MA_std' ,'ma_aver', 'ma_std']
        traits = []
        for i in range(len(category_name)):
            traits_category = traits_all(globals()[self.category_name[i]])
            traits_name_category = [self.category_name[i] + '_' + row for row in self.traits_name_all]
            traits_name.extend(traits_name_category)
            traits.extend(traits_category)

        self.trait_name = traits_name
        self.trait = traits

    def trait_stele_section_entire_hull(self):
        trait_name_entire = ['width', 'height', 'aspect_ratio', 'area', 'equi_diameter', 'perimeter', 'radius', 'MA', 'ma']
        self.trait_stele_entire_name = ['stele_entire_' + row for row in trait_name_entire]
        self.traits_stele_entire = traits_cul(self.contour_stele)[0]
        self.trait_section_entire_name = ['section_entire_' + row for row in trait_name_entire]
        self.traits_section_entire = traits_cul(self.contour_section)[0]
        self.trait_stele_hull_name = ['stele_hull_' + row for row in trait_name_entire]
        self.traits_stele_hull = traits_cul(self.hull_stele)[0]
        self.trait_section_hull_name = ['section_hull_' + row for row in trait_name_entire]
        self.traits_section_hull = traits_cul(self.hull_section)[0]

    def trait_multi_class(self):
        traits_cortex_all = []
        traits_stele_all = []
        for cell in self.cell_annotation['annotations']:
            if cell['category_name'] in ['endodermis', 'cortex']:
                traits_cortex_all.append(traits_cul(np.array(cell['contours'])))
            elif cell['category_name'] in ['pericycle', 'mexylem', 'stele']:
                traits_stele_all.append(traits_cul(np.array(cell['contours'])))

        self.traits_cortex_all_name = ['cortex_all_cell' + '_' + row for row in self.traits_name_all]
        self.traits_stele_all_name = ['stele_all_cell' + '_' + row for row in self.traits_name_all]
        self.traits_cortex_all = traits_all(traits_cortex_all)
        self.traits_stele_all = traits_all(traits_stele_all)


    def trait_ratio(self):
        self.ratio_traits_name = ['SWR', 'SCSAR', 'CCCAR', 'WCWAR', 'SCR', 'SCCCR', 'CNSNR', 'CFNSR']
        self.ratio_traits = []
        area_cortex_entire = self.traits_section_entire[3] - self.traits_stele_entire[3]

        SWR = self.traits_stele_entire[3] / self.traits_section_entire[3]
        SCSAR = self.traits_stele_all[2] / self.traits_stele_entire[3]
        CCCAR = self.traits_cortex_all[2] / area_cortex_entire
        WCWAR = (self.traits_stele_all[2] + self.traits_cortex_all[2]) / self.traits_section_entire[3]
        SCR = self.traits_stele_entire[3] / area_cortex_entire
        SCCCR = self.traits_stele_all[2] / self.traits_cortex_all[2]

        if self.traits_stele_all[0] != 0:
            CNSNR = self.traits_cortex_all[0] / self.traits_stele_all[0]
        else:
            CNSNR = 0

        if self.traits_stele_all[1] != 0:
            CFNSR = self.traits_cortex_all[1] / self.traits_stele_all[1]
        else:
            CFNSR = 0

        self.ratio_traits = [SWR, SCSAR, CCCAR, WCWAR, SCR, SCCCR, CNSNR, CFNSR]


    def trait_all_flow(self):
        self.preprocess()
        self.cell_detection()
        self.cell_class()
        self.trait_cal()
        self.trait_stele_section_entire_hull()
        self.trait_multi_class()
        self.trait_ratio()

        self.trait_name.extend(self.trait_stele_entire_name)
        self.trait.extend(self.traits_stele_entire)
        self.trait_name.extend(self.trait_section_entire_name)
        self.trait.extend(self.traits_section_entire)
        self.trait_name.extend(self.trait_stele_hull_name)
        self.trait.extend(self.traits_stele_hull)
        self.trait_name.extend(self.trait_section_hull_name)
        self.trait.extend(self.traits_section_hull)
        self.trait_name.extend(self.traits_cortex_all_name)
        self.trait.extend(self.traits_cortex_all)
        self.trait_name.extend(self.traits_stele_all_name)
        self.trait.extend(self.traits_stele_all)
        self.trait_name.extend(self.ratio_traits_name)
        self.trait.extend(self.ratio_traits)

        return self.cell_annotation, self.cell_area, self.trait_name, self.trait, self.img_last
