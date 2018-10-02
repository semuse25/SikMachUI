# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import yaml
from keras import backend as K
import neuralgym as ng
import core
import seg_model
from inpaint_model import InpaintCAModel


class Ui_MainWindow(object):
    def init(self):
        self.image = None
        self.origin = None
        self.mask = None
        self.back = None
        self.imgList = []
        self.maskList = []
        self.doneList = []
        self.fileIndex = 0
        self.showMaskMode = False
        self.showResultMode = False
        self.penColor = (0,0,255)
        self.penSize = 10
        self.mode = 'PEN'
        self.saveName = ''
        self.segmap = None
##
##        print("Start Loading....")
##        #TODO: 실행 전에 터지지 않는 적절한 값으로 딱 한번 초기화 할 것.
##        core.seg_limit   = 4000000 # 보통 이게 더 큼
##        core.compl_limit = 1492400 #
##
##        segnet_yml = 'segnet/seg48_4[553].yml' # segnet configuration
##        segnet_model_path = 'segnet/seg48_4[553].h5' # saved segnet model
##
##        complnet_ckpt_dir = 'v2_180923' # saved complnets directory
##        #--------------------------------------------
##        # for segnet
##        with open(segnet_yml,'r') as f:
##            self.config = yaml.load(f)
##
##        # for complnet
##        self.complnet = InpaintCAModel('v2')
##
##        ng.get_gpus(1,False) #TODO: 단 한 번만 호출할 것.
##        #--------------------------------------------
##        self.dilate_kernel = core.rect5
##        print("Loading Complete")


    def setupUi(self, MainWindow):
        self.init()
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280,680)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(0, 0, 999, 649))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 797, 597))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label.setGeometry(QtCore.QRect(5, 44, 491, 451))
        self.label.setObjectName("label")
        self.label.setMouseTracking(True)
        self.scrollArea.setWidget(self.label)
        MainWindow.setCentralWidget(self.centralwidget)

        #menu bar item
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.toolbar1 = QtWidgets.QToolBar(MainWindow)
        self.toolbar1.setGeometry(QtCore.QRect(1000, 22, 250, 30))
        self.toolbar1.setObjectName("toolbar1")
        self.toolbar2 = QtWidgets.QToolBar(MainWindow)
        self.toolbar2.setGeometry(QtCore.QRect(1000, 52, 250, 30))
        self.toolbar2.setObjectName("toolbar2")
        self.toolbar3 = QtWidgets.QToolBar(MainWindow)
        self.toolbar3.setGeometry(QtCore.QRect(1000, 82, 250, 30))
        self.toolbar3.setObjectName("toolbar3")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        MainWindow.setMenuBar(self.menubar)
        self.actionopen_file = QtWidgets.QAction(MainWindow)
        self.actionopen_file.setObjectName("actionopen_file")
        self.actionopen_file.triggered.connect(self.getImageFile)
        self.actionopen_folder = QtWidgets.QAction(MainWindow)
        self.actionopen_folder.setObjectName("actionopen_folder")
        self.actionopen_folder.triggered.connect(self.getImageFolder)
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave.triggered.connect(self.saveImage)
        self.actionprogram_information = QtWidgets.QAction(MainWindow)
        self.actionprogram_information.setObjectName("actionprogram_information")
        self.menu.addAction(self.actionopen_file)
        self.menu.addAction(self.actionopen_folder)
        self.menu.addAction(self.actionSave)
        self.menu_3.addAction(self.actionprogram_information)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)


        #toolbar item
        self.actionPrevFile = QtWidgets.QAction(QtGui.QIcon("icon/prev.png"),"이전 페이지",self.toolbar1)
        self.actionPrevFile.setObjectName("actionPrevFile")
        self.actionPrevFile.triggered.connect(self.prevPage)
        self.actionNextFile = QtWidgets.QAction(QtGui.QIcon("icon/next.png"),"다음 페이지",self.toolbar1)
        self.actionNextFile.setObjectName("actionNextFile")
        self.actionNextFile.triggered.connect(self.nextPage)
        self.numOfCurrentPage = QtWidgets.QLineEdit(self.toolbar1)
        self.numOfCurrentPage.setObjectName("numOfCurrentPage")
        self.numOfCurrentPage.setMaxLength(3)
        self.numOfCurrentPage.resize(30,30)
        self.numOfPage = QtWidgets.QLabel(self.toolbar1)
        self.numOfPage.setObjectName("numOfPage")
        self.pageJumpBtn = QtWidgets.QPushButton(self.toolbar1)
        self.pageJumpBtn.setText('이동')
        self.pageJumpBtn.resize(20,20)
        self.pageJumpBtn.clicked.connect(self.pageJump)
        self.autoBtn = QtWidgets.QAction(QtGui.QIcon("icon/auto.png"),"자동 처리",self.toolbar2)
        self.autoBtn.setObjectName("actionAutoBtn")
        self.autoBtn.triggered.connect(self.auto)
        self.segNetBtn = QtWidgets.QAction(QtGui.QIcon("icon/text.png"),"글자 영역 추출",self.toolbar2)
        self.segNetBtn.setObjectName("actionSegNetBtn")
        self.segNetBtn.triggered.connect(self.segNet)
        self.complNetBtn = QtWidgets.QAction(QtGui.QIcon("icon/textClean.png"),"리드로잉",self.toolbar2)
        self.complNetBtn.setObjectName("actionComplNetBtn")
        self.complNetBtn.triggered.connect(self.ComplNet)
        self.showMaskBtn = QtWidgets.QAction(QtGui.QIcon("icon/maskOn.png"),"마스크 보기",self.toolbar3)
        self.showMaskBtn.setObjectName("actionShowMaskBtn")
        self.showMaskBtn.triggered.connect(self.showMaskToggle)
        self.rectEraser = QtWidgets.QAction(QtGui.QIcon("icon/rect.png"),"사각형 영역 지우기",self.toolbar3)
        self.rectEraser.setObjectName("actionRectEraser")
        self.rectEraser.triggered.connect(self.eraseRect)
        self.penEraser = QtWidgets.QAction(QtGui.QIcon("icon/eraser.png"),"지우개",self.toolbar3)
        self.penEraser.setObjectName("actionPenEraser")
        self.penEraser.triggered.connect(self.erasePen)
        self.redPen = QtWidgets.QAction(QtGui.QIcon("icon/redPen.png"),"red 마스크 그리기",self.toolbar3)
        self.redPen.setObjectName("actionRedPen")
        self.redPen.triggered.connect(self.drawRed)
##        self.bluePen = QtWidgets.QAction(QtGui.QIcon("icon/bluePen.png"),"blue 마스크 그리기",self.toolbar3)
##        self.bluePen.setObjectName("actionBluePen")
##        self.bluePen.triggered.connect(self.drawBlue)
        self.actionPenMinus = QtWidgets.QAction(QtGui.QIcon("icon/minus.png"),"펜 크기 줄이기",self.toolbar3)
        self.actionPenMinus.setObjectName("actionPrevFile")
        self.actionPenMinus.triggered.connect(self.penSizeDown)
        self.actionPenPlus = QtWidgets.QAction(QtGui.QIcon("icon/plus.png"),"펜 크기 늘리기",self.toolbar3)
        self.actionPenPlus.setObjectName("actionNextFile")
        self.actionPenPlus.triggered.connect(self.penSizeUP)
        self.sizeVal = QtWidgets.QLabel(self.toolbar3)
        self.sizeVal.setObjectName("numOfPage")
        self.sizeVal.setText('10')
        self.toolbar1.addAction(self.actionPrevFile)
        self.toolbar1.addWidget(self.numOfCurrentPage)
        self.toolbar1.addWidget(self.numOfPage)
        self.toolbar1.addWidget(self.pageJumpBtn)
        self.toolbar1.addAction(self.actionNextFile)
        #self.toolbar.addSeparator()
        self.toolbar2.addAction(self.autoBtn)
        self.toolbar2.addAction(self.segNetBtn)
        self.toolbar2.addAction(self.complNetBtn)
        #self.toolbar.addSeparator()
        self.toolbar3.addAction(self.showMaskBtn)
        self.toolbar3.addAction(self.penEraser)
        self.toolbar3.addAction(self.rectEraser)
        self.toolbar3.addAction(self.redPen)
##        self.toolbar3.addAction(self.bluePen)
        self.toolbar3.addAction(self.actionPenMinus)
        self.toolbar3.addWidget(self.sizeVal)
        self.toolbar3.addAction(self.actionPenPlus)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menu.setTitle(_translate("MainWindow", "파일"))
        self.menu_2.setTitle(_translate("MainWindow", "처리"))
        self.menu_3.setTitle(_translate("MainWindow", "도움말"))
        self.actionopen_file.setText(_translate("MainWindow", "파일 열기"))
        self.actionopen_folder.setText(_translate("MainWindow", "폴더 열기"))
        self.actionSave.setText(_translate("MainWindow", "저장"))
        self.actionprogram_information.setText(_translate("MainWindow", "프로그램 정보"))
        self.numOfPage.setText("/ 0")
        self.numOfCurrentPage.setText('0')

##    def bgr_float32(self,uint8img):
##        c = 1 if len(uint8img.shape) == 2 else 3
##        h,w = uint8img.shape[:2]
##        uint8img = (uint8img / 255).astype(np.float32)
##        return uint8img.reshape((h,w,c))


    def prevPage(self):
        if self.fileIndex > 0:
            self.fileIndex -= 1
            self.numOfCurrentPage.setText(str(self.fileIndex+1))
            self.image = cv2.imread(self.imgList[self.fileIndex])
            if len(self.maskList) != 0:
                self.origin = self.maskList[self.fileIndex]
                self.mask = self.origin.copy()
            self.showImage()
        else:
            self.statusbar.showMessage('첫번째 페이지입니다.')

    def nextPage(self):
        if self.fileIndex < len(self.imgList)-1:
            self.fileIndex += 1
            self.numOfCurrentPage.setText(str(self.fileIndex+1))
            self.image = cv2.imread(self.imgList[self.fileIndex])
            if len(self.maskList) != 0:
                self.origin = self.maskList[self.fileIndex]
                self.mask = self.origin.copy()
            self.showImage()
        else:
            self.statusbar.showMessage('마지막 페이지입니다.')

    def pageJump(self):
        if int(self.numOfCurrentPage.text()) in range(1,len(self.imgList)+1):
            self.fileIndex = int(self.numOfCurrentPage.text())-1
            self.image = cv2.imread(self.imgList[self.fileIndex])
            if len(self.maskList) != 0:
                self.origin = self.maskList[self.fileIndex]
                self.mask = self.origin.copy()
            self.showImage()
        else:
            self.statusbar.showMessage('잘못된 페이지 번호입니다.')

    def getImageFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Buscar Imagen', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if filename:
            self.imgList = []
            self.maskList = []
            self.saveName = filename.split('/')[-1]
            self.imgList.append(filename)
            self.fileIndex = 0
            self.image = cv2.imread(self.imgList[self.fileIndex])
            self.mask = np.zeros(self.image.shape,np.uint8)
            self.maskList.append(self.mask)
            self.doneList.append(None)
            self.origin = self.mask.copy()
            self.showImage()
            self.numOfPage.setText('/ 1')
            self.numOfCurrentPage.setText('1')

    def getImageFolder(self):
        folderName = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))
        self.imgList = []
        self.maskList = []
        self.doneList = []
        self.saveName = folderName.split('/')[-1]
        for root, dirs, files in os.walk(folderName):
            for fname in files:
                if fname.split('.')[-1] in ('png','jpg','jpeg','bmp'):
                    fileName = os.path.join(root, fname)
                    img = cv2.imread(fileName)
                    mask = np.zeros(img.shape,np.uint8)
                    self.maskList.append(mask)
                    self.imgList.append(fileName)
                    self.doneList.append(None)
        self.fileIndex = 0
        self.image = cv2.imread(self.imgList[self.fileIndex])
        self.mask = self.maskList[self.fileIndex].copy()
        self.origin = self.mask.copy()
        self.showImage()
        self.numOfPage.setText('/ ' + str(len(self.imgList)))
        self.numOfCurrentPage.setText('1')

    def showImage(self):
        size = self.image.shape
        step = self.image.size / size[0]
        qformat = QtGui.QImage.Format_Indexed8

        if len(size) == 3:
            if size[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888

        if self.showResultMode is True:
            img = QtGui.QImage(self.doneList[self.fileIndex], size[1], size[0], step, qformat)
        elif self.showMaskMode is False:
            img = QtGui.QImage(self.image, size[1], size[0], step, qformat)
        else:
            gmask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
            fmask = cv2.threshold(gmask, 10, 255, cv2.THRESH_BINARY)[1]
            bmask = cv2.bitwise_not(fmask)
            fg = cv2.bitwise_and(self.mask, self.mask, mask=fmask)
            bg = cv2.bitwise_and(self.image, self.image, mask=bmask)
            maskedImg = cv2.add(fg,bg)

            img = QtGui.QImage(maskedImg, size[1], size[0], step, qformat)
        img = img.rgbSwapped()

        self.label.setPixmap(QtGui.QPixmap.fromImage(img))
        #print(size)
        #self.resize(self.label.pixmap().size())
        #MainWindow.resize(size[1]+20,size[0]+16)
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(5, 44, size[1], size[0]))
        self.label.setGeometry(QtCore.QRect(5, 44, size[1], size[0]))

    def segNet(self):
        if self.showResultMode is False:
            img = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
            img = bgr_float32(img)
            self.segmap = load_segment_unload(img, config, segnet_model_path)
        self.segmap = (self.segmap >= 0.5).astype(np.uint8) * 255
        self.maskList[self.fileIndex] = np.zeros(self.image.shape,np.uint8)
        self.maskList[self.fileIndex][:,:,2] = self.segmap
        self.mask = self.maskList[self.fileIndex]
        self.origin = self.mask.copy()
        self.showMaskMode = True
        self.showResultMode = False
        self.showMaskBtn.setIcon(QtGui.QIcon("icon/maskOff.png"))
        self.showImage()

    def ComplNet(self):
        if self.segmap is None:
            return None
        img = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        img = bgr_float32(img)
        result = core.inpaint(img, segmap,
                      complnet, complnet_ckpt_dir,
                      dilate_kernel=dilate_kernel)
        self.doneList[self.fileIndex] = result
        self.image = result
        self.showResultMode = True
        self.showImage()
        self.showMaskMode = False
        self.showMaskBtn.setIcon(QtGui.QIcon("icon/maskOn.png"))

    def auto(self):
        self.segNet()
        self.ComplNet()

    def showMaskToggle(self):
        if self.showResultMode is False:
            if self.showMaskMode is True:
                self.showMaskMode = False
                self.showMaskBtn.setIcon(QtGui.QIcon("icon/maskOn.png"))
            else:
                self.showMaskMode = True
                self.showMaskBtn.setIcon(QtGui.QIcon("icon/maskOff.png"))

        self.showImage()

    def saveImage(self):
        if len(self.imgList) == 1:
            cv2.imwrite('./cleaned/' + self.saveName,self.doneList[0])
        else:
            if not os.path.isdir('./cleaned/'+self.saveName):
                os.mkdir('./cleaned/'+self.saveName)
            for i in range(len(self.maskList)):
                cv2.imwrite('./cleaned/' + self.saveName + '/' + str(i)+'.jpg',self.doneList[i])



    def eraseRect(self):
        self.mode = 'RECT_ERASE'
    def erasePen(self):
        self.mode = 'PEN_ERASE'
        self.penColor = (0,0,0)

    def drawRed(self):
        self.penColor = (0,0,255)
        self.mode = 'PEN'
    def drawBlue(self):
        self.penColor = (255,0,0)
        self.mode = 'PEN'

    def penSizeUP(self):
        if self.penSize < 30:
            self.penSize += 1
            self.sizeVal.setText(str(self.penSize))

    def penSizeDown(self):
        if self.penSize > 1:
            self.penSize -= 1
            self.sizeVal.setText(str(self.penSize))

class MyMainScreen(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.label.setMouseTracking(True)
        self.ui.label.installEventFilter(self)
        self.x = 0
        self.y = 0
        self.drawing = False

    def eventFilter(self, source, event):
        if self.ui.mode == 'RECT_ERASE' and self.ui.showMaskMode == True:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self.drawing = True
                self.x = event.pos().x()
                self.y = event.pos().y()
            if event.type() == QtCore.QEvent.MouseMove and self.drawing is True:
                self.ui.mask = self.ui.origin.copy()
                cv2.rectangle(self.ui.mask,(self.x,self.y),(event.pos().x(),event.pos().y()),(255,0,0),1)
                self.ui.showImage()
                #print(event.pos().x())
            if event.type() == QtCore.QEvent.MouseButtonRelease and self.drawing is True:
                self.drawing = False
                self.ui.mask = self.ui.origin.copy()
                self.ui.showImage()
                if self.x < event.pos().x():
                    x1 = self.x
                    x2 = event.pos().x()
                else:
                    x1 = event.pos().x()
                    x2 = self.x
                if self.y < event.pos().y():
                    y1 = self.y
                    y2 = event.pos().y()
                else:
                    y1 = event.pos().y()
                    y2 = self.y

                if x1 < 0 :
                    x1 = 0
                if y1 < 0 :
                    y1 = 0
                if x2 > self.ui.image.shape[1] :
                    x2 = self.ui.image.shape[1]
                if y2 > self.ui.image.shape[0] :
                    y2 = self.ui.image.shape[0]



                self.ui.mask[y1:y2,x1:x2] = np.zeros((y2-y1,x2-x1,3),np.uint8)
                self.ui.back = self.ui.origin
                self.ui.origin = self.ui.mask.copy()
                self.ui.maskList[self.ui.fileIndex] = self.ui.mask.copy()
                self.ui.showImage()
        elif (self.ui.mode == 'PEN' or self.ui.mode == 'PEN_ERASE') and self.ui.showMaskMode == True:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self.drawing = True
                cv2.circle(self.ui.mask,(event.pos().x(),event.pos().y()),self.ui.penSize,self.ui.penColor,-1)
                self.ui.showImage()
            elif event.type() == QtCore.QEvent.MouseMove and self.drawing is True:
                cv2.circle(self.ui.mask,(event.pos().x(),event.pos().y()),self.ui.penSize,self.ui.penColor,-1)
                self.ui.showImage()
            elif event.type() == QtCore.QEvent.MouseMove and self.drawing is False:
                self.ui.mask = self.ui.origin.copy()
                cv2.circle(self.ui.mask,(event.pos().x(),event.pos().y()),self.ui.penSize,self.ui.penColor,-1)
                self.ui.showImage()
            elif event.type() == QtCore.QEvent.MouseButtonRelease and self.drawing is True:
                self.drawing = False
                self.ui.back = self.ui.origin
                self.ui.origin = self.ui.mask
                self.ui.maskList[self.ui.fileIndex] = self.ui.mask.copy()


        return QtWidgets.QMainWindow.eventFilter(self, source, event)



def bgr_float32(uint8img):
    c = 1 if len(uint8img.shape) == 2 else 3
    h,w = uint8img.shape[:2]
    uint8img = (uint8img / 255).astype(np.float32)
    return uint8img.reshape((h,w,c))

def load_segment_unload(img, config, segnet_model_path):
    # get configuration
    kernel_init = config.get('kernel_init')
    num_maxpool = config.get('num_maxpool')
    num_filters = config.get('num_filters')
    filter_vec = config.get('filter_vec')
    num_classes = config.get('num_classes')
    # load unet
    segnet = seg_model.unet(
        segnet_model_path, (None,None,1),
        kernel_init=kernel_init, num_classes=num_classes,
        last_activation = 'sigmoid' if num_classes == 2 else 'softmax',
        num_filters=num_filters, num_maxpool = num_maxpool,
        filter_vec=filter_vec)
    # segment the image
    segmap = core.segment(segnet, img)
    # unload unet
    K.clear_session()

    return segmap


if __name__ == "__main__":
    #TODO: 실행 전에 터지지 않는 적절한 값으로 딱 한번 초기화 할 것.
    core.seg_limit   = 4000000 // 10 # 보통 이게 더 큼
    core.compl_limit = 1492400 // 10 #

    segnet_yml = 'segnet/seg48_4[553].yml' # segnet configuration
    segnet_model_path = 'segnet/seg48_4[553].h5' # saved segnet model

    complnet_ckpt_dir = 'v2_180923' # saved complnets directory
    #--------------------------------------------
    # for segnet
    with open(segnet_yml,'r') as f:
        config = yaml.load(f)

    # for complnet
    complnet = InpaintCAModel('v2')

    ng.get_gpus(1,False) #TODO: 단 한 번만 호출할 것.
    #--------------------------------------------
    dilate_kernel = core.rect5


    if not os.path.isdir('./cleaned'):
        os.mkdir('./cleaned')
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MyMainScreen()
    MainWindow.show()
    #app.exec_()
    sys.exit(app.exec_())

