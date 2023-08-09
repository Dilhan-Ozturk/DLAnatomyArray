import os
import sys
import cv2 as cv
import numpy as np
import openpyxl as op

from ui.MainWindow import Ui_MainWindow
# from ImageOperator import *
from PyQt5.QtWidgets import QMainWindow, QStyle, QApplication, QFileDialog, QMessageBox,QGraphicsScene, QApplication, QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QPixmap,QImage, QWheelEvent, QIcon
from PyQt5.QtCore import QRect, QRectF, QSize, Qt,QThread,QEventLoop, QTimer
from PyQt5 import QtCore,QtGui
from DLSeg import DLsegBacth
from qt_material import apply_stylesheet
from ModelTrain import train_model
import json
from RootTraitCal import RootTraitCal
from ImageProcess import *


class SegThread(QThread):
    signal_img1_set = QtCore.pyqtSignal(np.ndarray)
    signal_img2_set = QtCore.pyqtSignal(np.ndarray)
    signal_process_set = QtCore.pyqtSignal(int)

    def __init__(self, filedirpath, weightpath, savepath):

        super(SegThread, self).__init__()
        self.filedirpath = filedirpath
        self.weightpath = weightpath
        self.savepath = savepath

    def __del__(self):
        self.wait()

    def run(self):
        file = os.listdir(self.filedirpath)
        i = 1
        for name in file:
            filepath = self.filedirpath + '\\' + name
            img = DLsegBacth(filepath, self.weightpath, name, self.savepath)
            img2 = cv.imread(filepath)
            processnumber = int(i/len(file)*100)
            i += 1
            img = img_resize(img, 1024)
            img2 = img_resize(img2, 1024)

            self.signal_img1_set.emit(img)
            self.signal_img2_set.emit(img2)

            self.signal_process_set.emit(processnumber)

            print(name + ' has been successfully segmented')

        print('All is done')


class TraitsThread(QThread):
    signal_result_set = QtCore.pyqtSignal(np.ndarray)
    signal_result2_set = QtCore.pyqtSignal(np.ndarray)
    signal_process2_set = QtCore.pyqtSignal(int)
    def __init__(self, filedirpath, savepath, DS_pericycle, DS2_pericycle, Area_pericycle, DS1_endodermis,
                                   DS2_endodermis, Area_endodermis, DS1_exodermis, DS2_exodermis, Area_exodermis):

        super(TraitsThread, self).__init__()
        self.filedirpath = filedirpath
        self.savepath = savepath
        self.DS_pericycle = DS_pericycle
        self.DS2_pericycle = DS2_pericycle
        self.Area_pericycle = Area_pericycle
        self.DS1_endodermis = DS1_endodermis
        self.DS2_endodermis = DS2_endodermis
        self.Area_endodermis = Area_endodermis
        self.DS1_exodermis = DS1_exodermis
        self.DS2_exodermis = DS2_exodermis
        self.Area_exodermis = Area_exodermis


    def __del__(self):
        self.wait()

    def run(self):
        wb = op.Workbook()  # 创建工作簿对象
        ws_trait = wb.create_sheet('traits')
        ws_stele = wb.create_sheet('stele')
        ws_mexylem = wb.create_sheet('mexylem')
        ws_pericycle = wb.create_sheet('pericycle')
        ws_endodermis = wb.create_sheet('endodermis')
        ws_cortex = wb.create_sheet('cortex')
        ws_exodermis = wb.create_sheet('exodermis')

        traits_name_index = ['name']


        path_seg = os.path.join(self.filedirpath,  'in')

        file = os.listdir(path_seg)
        i = 1

        if os.path.exists(self.savepath) is False:
            os.makedirs(self.savepath)

        for name in file:
            traits_all_index = [name]
            img_in_path = os.path.join(self.filedirpath, 'in', name)
            img_out_path = os.path.join(self.filedirpath, 'out', name)
            img_in = cv.imread(img_in_path, 0)
            img_out = cv.imread(img_out_path, 0)
            img_all = cv.add(img_in, img_out)
            processnumber2 = int(i / len(file) * 100)
            RootTrait = RootTraitCal(name, img_in, img_out, self.DS_pericycle, self.DS2_pericycle, self.Area_pericycle, self.DS1_endodermis,
                                   self.DS2_endodermis, self.Area_endodermis, self.DS1_exodermis, self.DS2_exodermis, self.Area_exodermis)

            try:
                cell_annotation, cell_area, trait_name,  trait, img_last = RootTrait.trait_all_flow()
                traits_name_index.extend(trait_name)

                if i == 1:
                    ws_trait.append(traits_name_index)

                traits_all_index.extend(trait)
                ws_trait.append(traits_all_index)

                area_pericyle = [name]
                area_mexylem = [name]
                area_stele = [name]
                area_endodermis = [name]
                area_exodermis = [name]
                area_cortex = [name]

                for area_index in range(len(cell_area['area'])):
                    if 'pericycle' in cell_area['area'][area_index].keys():
                        area_pericyle.append(cell_area['area'][area_index]['pericycle'])
                    elif 'mexylem' in cell_area['area'][area_index].keys():
                        area_mexylem.append(cell_area['area'][area_index]['mexylem'])
                    elif 'stele' in cell_area['area'][area_index].keys():
                        area_stele.append(cell_area['area'][area_index]['stele'])
                    elif 'endodermis' in cell_area['area'][area_index].keys():
                        area_endodermis.append(cell_area['area'][area_index]['endodermis'])
                    elif 'exodermis' in cell_area['area'][area_index].keys():
                        area_exodermis.append(cell_area['area'][area_index]['exodermis'])
                    elif 'cortex' in cell_area['area'][area_index].keys():
                        area_cortex.append(cell_area['area'][area_index]['cortex'])


                ws_pericycle.append(area_pericyle)
                ws_mexylem.append(area_mexylem)
                ws_stele.append(area_stele)
                ws_endodermis.append(area_endodermis)
                ws_exodermis.append(area_exodermis)
                ws_cortex.append(area_cortex)

                i += 1

                if os.path.exists(os.path.join(self.savepath , 'ImgResult')) is False:
                    os.makedirs(os.path.join(self.savepath , 'ImgResult'))

                if os.path.exists(os.path.join(self.savepath , 'jsonfile')) is False:
                    os.makedirs(os.path.join(self.savepath , 'jsonfile'))


                wb.save(os.path.join(self.savepath, 'result.xlsx'))
                cv.imwrite(os.path.join(self.savepath, 'ImgResult', name), img_last)
                filename = os.path.splitext(name)
                # json_data = json.dumps(cell_annotation)
                # with open(os.path.join(self.savepath, 'jsonfile', filename[0] + '.json'), 'w+', encoding='utf-8') as fp:
                #     fp.write(json_data)
                print(name + ' is done')

                img_last = img_resize(img_last, 1024)
                self.signal_result_set.emit(img_last)


            except:
                print(name + ' something wrong')

            img_all = img_resize(img_all, 1024)

            self.signal_result2_set.emit(img_all)
            self.signal_process2_set.emit(processnumber2 + 1)
        print('All is done')

class TrainThread(QThread):
    def __init__(self, datapath, savepath, LR,ITERS, BS, SI, LI, CLA):

        super(TrainThread, self).__init__()
        self.datapath = datapath
        self.savepath = savepath
        self.LR = LR
        self.ITERS = ITERS
        self.BS = BS
        self.SI = SI
        self.LI = LI
        self.CLA = CLA

    def __del__(self):
        self.wait()

    def run(self):
        train_model(self.datapath, self.savepath, self.LR, self.ITERS, self.BS, self.SI, self.LI, self.CLA)


class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str) #定义一个发送str的信号
    def write(self, text):
        self.textWritten.emit(str(text))


class UIMain(QMainWindow):

    def __init__(self):
        super().__init__()
        self.CellUI = Ui_MainWindow()
        self.CellUI.setupUi(self)
        self.setWindowIcon( QIcon('logo.png'))
        self.imgtest = 1
        self.imggray = 1
        self.weightpath = 1
        self.filedirpath = 1
        self.filedirpath2 = 1
        self.savepath = 'output/'
        self.Trainpath = 'data/'
        self.modelsavepath = 'output/'
        self.DLsavepath = 'output/'


        # self.filedirpath = 'D:/out_new/reference'
        # self.weightpath = 'weight/anatomyunet.pdparams'


        self.CellUI.pushButton_52.clicked.connect(self.open_weight)
        self.CellUI.pushButton_51.clicked.connect(self.deeplearning_seg)
        # self.CellUI.pushButton_55.clicked.connect(self.Traits_cul)
        self.CellUI.pushButton_24.clicked.connect(self.choose_filedir)
        self.CellUI.pushButton_54.clicked.connect(self.choose_filedir2)
        self.CellUI.pushButton_55.clicked.connect(self.choose_filedir3)
        self.CellUI.pushButton_57.clicked.connect(self.get_result)
        self.CellUI.pushButton_26.clicked.connect(self.choose_filedir4)
        self.CellUI.pushButton_27.clicked.connect(self.choose_filedir5)
        self.CellUI.pushButton_25.clicked.connect(self.choose_filedir6)
        self.CellUI.pushButton_28.clicked.connect(self.train)

        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stdin = EmittingStr(textWritten=self.outputWritten)

        self.CellUI.lineEdit.setText('0.1')
        self.CellUI.lineEdit_2.setText('20000')
        self.CellUI.lineEdit_3.setText('1')
        self.CellUI.lineEdit_4.setText('100')
        self.CellUI.lineEdit_5.setText('10')
        self.CellUI.lineEdit_6.setText('3')
        # self.CellUI.lineEdit_7.setText('20')
        # self.CellUI.lineEdit_8.setText('8')

    def outputWritten(self, text):
        cursor = self.CellUI.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.CellUI.textBrowser.setTextCursor(cursor)
        self.CellUI.textBrowser.ensureCursorVisible()


    def process_bar1(self,pv):
        self.CellUI.progressBar.setMinimum(0)
        self.CellUI.progressBar.setMaximum(100)
        self.CellUI.progressBar.setValue(pv)


    def show_img(self,image):

        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2RGB)
        elif image.shape[2] == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        else:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        graph_scene = QGraphicsScene()
        self.CellUI.graphicsView_2.setScene(graph_scene)

        if isinstance(image, str):
            image = QPixmap(image)
        elif isinstance(image, QPixmap):
            pass
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    image = QPixmap(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888))
            else:
                image = QPixmap(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8))
        elif len(image.shape) == 2:
            image = QPixmap(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8))
        else:
            raise TypeError('image type not supported')
        image = image.scaled(self.CellUI.graphicsView_2.width() - 3, self.CellUI.graphicsView_2.height() - 3,
                             QtCore.Qt.KeepAspectRatio)
        graph_scene.addPixmap(image)
        graph_scene.update()

    def show_img2(self,image):

        # 将任意格式的图片转成RGB三通道
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2RGB)
        elif image.shape[2] == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        else:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        graph_scene = QGraphicsScene()
        self.CellUI.graphicsView.setScene(graph_scene)

        if isinstance(image, str):
            image = QPixmap(image)
        elif isinstance(image, QPixmap):
            pass
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    image = QPixmap(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888))
            else:
                image = QPixmap(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8))
        elif len(image.shape) == 2:
            image = QPixmap(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8))
        else:
            raise TypeError('image type not supported')
        # 图片尺寸自适应
        image = image.scaled(self.CellUI.graphicsView.width() - 3, self.CellUI.graphicsView.height() - 3,
                             QtCore.Qt.KeepAspectRatio)


        graph_scene.addPixmap(image)
        graph_scene.update()

    def warning(self):
        title = 'warning'
        info = "Please select image file"
        message1 = QMessageBox.warning(self, title, info, QMessageBox.Yes |QMessageBox.Cancel)
        return message1

    def open_img(self):
        m = QFileDialog.getOpenFileName(None, "选取文件夹", ".")
        filepath = m[0]
        while (filepath.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')) == False):
            message1 = self.warning()
            print(message1)
            if message1 == 4194304:
                break
            m = QFileDialog.getOpenFileName(None, "choose_file", ".")
            filepath = m[0]
            continue
        print('The image filepath is ' + filepath)
        self.imgtest = cv.imread(filepath)
        self.show_img(self.imgtest)


    def open_weight(self):
        m = QFileDialog.getOpenFileName(None, "choose_weight", ".")
        filepath = m[0]
        while (filepath.lower().endswith(('.pdparams')) == False):
            message1 = self.warning()
            print(message1)
            if message1 == 4194304:
                break
            m = QFileDialog.getOpenFileName(None, "choose_file", ".")
            filepath = m[0]
            continue
        self.weightpath = filepath
        print('The weight path is ' + filepath)

    def deeplearning_seg(self):
        print('Begin to seg')
        self.thread = SegThread(self.filedirpath,self.weightpath, self.DLsavepath)
        self.thread.signal_img1_set.connect(self.show_img2)
        self.thread.signal_img2_set.connect(self.show_img)
        self.thread.signal_process_set.connect(self.process_bar1)
        self.thread.start()


    def choose_filedir(self):
        path = QFileDialog.getExistingDirectory(None, "请选择文件夹路径", ".")
        self.filedirpath = path
        print('The filedir path is ' + path)

    def choose_filedir2(self):
        path = QFileDialog.getExistingDirectory(None, "请选择文件夹路径", ".")
        self.filedirpath2 = path
        print('The segmented image filedir path is ' + path)

    def choose_filedir3(self):
        path = QFileDialog.getExistingDirectory(None, "请选择文件夹路径", ".")
        self.savepath = path
        print('The save path is ' + path)

    def choose_filedir4(self):
        path = QFileDialog.getExistingDirectory(None, "请选择文件夹路径", ".")
        self.Trainpath = path
        print('The data path is ' + path)

    def choose_filedir5(self):
        path = QFileDialog.getExistingDirectory(None, "请选择文件夹路径", ".")
        self.modelsavepath = path
        print('The model save path is ' + path)

    def choose_filedir6(self):
        path = QFileDialog.getExistingDirectory(None, "请选择文件夹路径", ".")
        self.DLsavepath = os.path.join(path, 'output')
        print('The predicted image save path is ' + path)

    def get_result(self):
        print('Begin to caculate traits')
        DS_pericycle = float(self.CellUI.lineEdit_8.text())
        DS2_pericycle = float(self.CellUI.lineEdit_9.text())
        Area_pericycle = float(self.CellUI.lineEdit_10.text())
        DS1_endodermis = float(self.CellUI.lineEdit_11.text())
        DS2_endodermis = float(self.CellUI.lineEdit_12.text())
        Area_endodermis = float(self.CellUI.lineEdit_13.text())
        DS1_exodermis = float(self.CellUI.lineEdit_14.text())
        DS2_exodermis = float(self.CellUI.lineEdit_15.text())
        Area_exodermis = float(self.CellUI.lineEdit_16.text())


        self.thread = TraitsThread(self.filedirpath2,self.savepath,
                                   DS_pericycle, DS2_pericycle, Area_pericycle, DS1_endodermis,
                                   DS2_endodermis, Area_endodermis, DS1_exodermis, DS2_exodermis, Area_exodermis)
        self.thread.signal_result_set.connect(self.show_img2)
        self.thread.signal_result2_set.connect(self.show_img)
        self.thread.signal_process2_set.connect(self.process_bar1)
        self.thread.start()


    def train(self):

        LR = self.CellUI.lineEdit.text()
        ITERS = self.CellUI.lineEdit_2.text()
        BS = self.CellUI.lineEdit_3.text()
        SI = self.CellUI.lineEdit_4.text()
        LI = self.CellUI.lineEdit_5.text()
        CLA = self.CellUI.lineEdit_6.text()

        print('Begin to train model')
        self.thread = TrainThread(self.Trainpath, self.modelsavepath, LR,ITERS, BS, SI, LI, CLA)
        self.thread.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    cell_ui = UIMain()
    apply_stylesheet(app, theme='dark_blue.xml')
    cell_ui.show()
    sys.exit(app.exec_())