from ImageProcess import *

class SegWithoutDL:
    def __init__(self, img_org, seg_thresh_section, seg_thresh_stele):
        self.img_org = img_org
        self.seg_thresh_section = seg_thresh_section
        self.seg_thresh_stele = seg_thresh_stele

    def preprocess(self):
        img_pro = cv.medianBlur(self.img_org, 3)
        # Divide the section area first
        _, thresh_img_section = cv.threshold(img_pro, self.seg_thresh_section, 255, cv.THRESH_BINARY)
        thresh_img_section = open_demo(thresh_img_section)
        thresh_img_section = FillHole(thresh_img_section)
        thresh_img_section = Find_max_region(thresh_img_section)
        self.mask_section = get_mask(img_pro, thresh_img_section)

        # Divide the stele area
        _, thresh_img_stele = cv.threshold(self.mask_section, self.seg_thresh_stele, 255, cv.THRESH_BINARY)
        thresh_img_stele = FillHole(thresh_img_stele)
        thresh_img_stele = Find_max_region(thresh_img_stele)
        self.mask_stele = get_mask(img_pro, thresh_img_stele)

        self.mask_cortex = cv.subtract(self.mask_section, self.mask_stele)

    def seg_stele_cortex(self):
        self.preprocess()

        # Segmentation of stele
        gray_lap_stele = cv.equalizeHist(self.mask_stele)
        # Adjust the adaptive parameters
        gray_lap_stele = cv.adaptiveThreshold(gray_lap_stele, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -5)
        gray_lap_stele = remove_small_objects(gray_lap_stele, 100)

        # Segmentation of cortex
        gray_lap_cortex = cv.equalizeHist(self.mask_cortex)
        # Adjust the adaptive parameters
        gray_lap_cortex = cv.adaptiveThreshold(gray_lap_cortex, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, -9)
        gray_lap_cortex = remove_small_objects(gray_lap_cortex, 1000)

        return gray_lap_stele, gray_lap_cortex


img_org = cv.imread('data/1.jpg', 0)

# Adjust these two parameters to separate the entire slice and stele area of the image
seg_thresh_section = 20
seg_thresh_stele = 60

SegWithoutDL = SegWithoutDL(img_org, seg_thresh_section, seg_thresh_stele)

img_stele, img_cortex = SegWithoutDL.seg_stele_cortex()