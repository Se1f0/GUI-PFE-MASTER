import sys
import time
from tkinter import Image
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QFileDialog, QMainWindow, QTableWidgetItem, QTableWidget
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic
from tools import *
from prepare import preprocessScan

class WelcomeScreen(QDialog):
    def __init__(self):
        super(WelcomeScreen,self).__init__()

        uic.loadUi("UI files/welcomeScreen.ui",self)
        
        self.getStartedButton.clicked.connect(self.goToNextScreen)
    
    def goToNextScreen(self):
        upload = UploadScreen()
        widget.addWidget(upload)
        widget.setCurrentIndex(widget.currentIndex()+1)

class UploadScreen(QDialog):
    def __init__(self):
        super(UploadScreen,self).__init__()
        uic.loadUi("UI files/uploadScreen.ui",self)

        self.mhdUploadButton.clicked.connect(self.openMhd)
        self.dicomUploadButton.clicked.connect(self.openDcm)
    
    def openMhd(self):
        fname = QFileDialog.getOpenFileName(self, "Open CT scan", "E:/PFE/LUNA/allset", "Mhd files (*.mhd)")
        if fname[0]:
            scanView = ScanViwerScreen(fname[0])
            widget.addWidget(scanView)
            widget.setCurrentIndex(widget.currentIndex()+1)
    
    def openDcm(self):
        dname = QFileDialog.getExistingDirectory(self, "Open CT scan", "E:/PFE/DSB3", QFileDialog.ShowDirsOnly)
        if dname:
            scanView = ScanViwerScreen(dname)
            widget.addWidget(scanView)
            widget.setCurrentIndex(widget.currentIndex()+1)

class ScanViwerScreen(QMainWindow):
    def __init__(self,path):
        super(ScanViwerScreen,self).__init__()

        uic.loadUi("UI files/scanViwerScreen.ui",self)

        self.loadLoadingScreen()

        self.path = path
        self.innitIndex = -1
        self.labelImage = None

        self.threadLoad = ThreadClass(self.path,1,parent=None)
        self.threadLoad.start()
        self.threadLoad.any_signal.connect(self.getThreadResults)

        self.threadPlay = None

    def getThreadResults(self,data):
        self.labelImage,self.innitIndex = data

        self.loading.setVisible(False)
        self.loadingLabel.setVisible(False)

        self.toolBar.setVisible(True)
        self.metaTitle.setVisible(True)
        self.metaDataTable.setVisible(True)
        self.goToPreprocessButton.setVisible(True)
        self.frame.setVisible(True)
        self.image.setVisible(True)
        self.imageName.setVisible(True)
        self.index.setVisible(True)
        self.verticalSlider.setVisible(True)
        
        self.loadInfos()

        self.actionNext_Slice.triggered.connect(self.slideRight)
        self.actionLeft_Slice.triggered.connect(self.slideLeft)
        self.actionReset.triggered.connect(self.reset)
        self.actionGoBack.triggered.connect(self.goBack)
        self.verticalSlider.setMinimum(0)
        self.verticalSlider.setMaximum(self.labelImage.shape[0]-1)
        self.verticalSlider.valueChanged.connect(self.slide)
        self.goToPreprocessButton.clicked.connect(self.goToPreprocessed)
        self.actionPlay.triggered.connect(self.play)
        self.actionpause.triggered.connect(self.pause)
        print("end")

    def play(self):
        self.actionpause.setEnabled(True)
        self.actionPlay.setEnabled(False)
        self.actionNext_Slice.setEnabled(False)
        self.actionLeft_Slice.setEnabled(False)
        self.actionReset.setEnabled(False)
        self.verticalSlider.setEnabled(False)
        self.verticalSlider.setEnabled(False)
        self.actionGoBack.setEnabled(False)
        self.threadPlay = PlayThread(parent=None)
        self.threadPlay.start()
        self.threadPlay.any_signal.connect(self.getPlayThreadResults)

    def getPlayThreadResults(self,isReady):
        self.slideRight()

    def pause(self):
        self.threadPlay.stop()
        self.actionpause.setEnabled(False)
        self.actionPlay.setEnabled(True)
        self.actionNext_Slice.setEnabled(True)
        self.actionLeft_Slice.setEnabled(True)
        self.actionReset.setEnabled(True)
        self.verticalSlider.setEnabled(True)
        self.goToPreprocessButton.setEnabled(True)
        self.actionGoBack.setEnabled(True)

    def loadLoadingScreen(self):
        self.toolBar.setVisible(False)
        self.metaTitle.setVisible(False)
        self.metaDataTable.setVisible(False)
        self.goToPreprocessButton.setVisible(False)
        self.frame.setVisible(False)
        self.image.setVisible(False)
        self.imageName.setVisible(False)
        self.index.setVisible(False)
        self.verticalSlider.setVisible(False)


        self.loadingLabel.setText("Loading")
        self.loadingLabel.adjustSize()
        movie = QtGui.QMovie('assets\loading.gif')
        self.loading.setMovie(movie)
        movie.start()

    def goToPreprocessed(self):
        preScanView = ProcessedScanViwerScreen(self.path)
        widget.addWidget(preScanView)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def updateImage(self):
        pixmap = QImage(self.labelImage[self.innitIndex], self.labelImage.shape[2], self.labelImage.shape[1], self.labelImage.shape[2]*3, QImage.Format_RGB888)
        self.image.setPixmap(QPixmap.fromImage(pixmap))
        self.verticalSlider.setValue(self.innitIndex)
        self.index.setText(f'{self.innitIndex:03}/{self.labelImage.shape[0]-1}')

    def slide(self,value):
        self.innitIndex = value
        self.updateImage()

    def goBack(self):
        widget.removeWidget(self)

    def slideRight(self):
        self.innitIndex += 1
        if(self.innitIndex >= self.labelImage.shape[0]):
            self.innitIndex = 0
        self.updateImage()
    
    def slideLeft(self):
        self.innitIndex = max(0,self.innitIndex-1)
        self.updateImage()
    
    def reset(self):
        self.innitIndex = 0
        self.updateImage()
    
    def loadInfos(self):
        self.updateImage()
        self.imageName.setText(self.path.split("/")[-1])
        self.imageName.adjustSize()
        self.imageName.setGeometry((830-self.imageName.size().width())//2,80,self.imageName.size().width(),self.imageName.size().height())
        self.setTableWidget()
        
    def setTableWidget(self):
        if ".mhd" in self.path:
            data = getMetaDataMhd(self.path)
        else:
            data = getMetaDataDcm(self.path)

        colPosition = self.metaDataTable.columnCount()
        self.metaDataTable.insertColumn(colPosition)
        self.metaDataTable.setHorizontalHeaderLabels(["Values"])

        header = self.metaDataTable.horizontalHeader()       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        for i in range(len(list(data.keys()))):
            rowPosition = self.metaDataTable.rowCount()
            self.metaDataTable.insertRow(rowPosition)
        self.metaDataTable.setVerticalHeaderLabels(list(data.keys()))

        for i in range(len(list(data.keys()))):
            self.metaDataTable.setItem(i , 0, QTableWidgetItem(data[list(data.keys())[i]]))

        self.metaDataTable.setFont(QtGui.QFont('Verdana',8))
        self.metaDataTable.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metaDataTable.resizeColumnsToContents()

class ThreadClass(QtCore.QThread):
    any_signal = QtCore.pyqtSignal(object)
    def __init__(self, path,screen,parent=None):
        super(ThreadClass,self).__init__(parent)
        self.path = path
        self.screen = screen
    
    def run(self):
        print("starting")
        if self.screen == 1:
            scan,index = loadImage(self.path)
            self.any_signal.emit((scan,index))
        else:
            scan = preprocessScan(self.path)
            self.any_signal.emit(scan)

class PlayThread(QtCore.QThread):
    any_signal = QtCore.pyqtSignal(object)
    def __init__(self,parent=None):
        super(PlayThread,self).__init__(parent)
        self.is_running = True
    
    def run(self):
        print("starting")
        t0 = int(round(time.time() * 1000))
        fps = 60
        updateFrame = 1000//fps
        isReady = False
        while True:
            t1 = int(round(time.time() * 1000))
            t2 = t1 - t0
            if t2 >= updateFrame:
                isReady = True
                t0 = int(round(time.time() * 1000))
                time.sleep(0.01)
                self.any_signal.emit(isReady)

    def stop(self):
        self.is_running = False
        self.terminate()
        print("stop")

class ProcessedScanViwerScreen(QMainWindow):
    def __init__(self,path):
        super(ProcessedScanViwerScreen,self).__init__()

        uic.loadUi("UI files/processedScanViwerScreen.ui",self)

        self.loadLoadingScreen()

        self.path = path
        self.innitIndex = 0
        self.im = None
        
        self.threadLoad = ThreadClass(self.path,2,parent=None)
        self.threadLoad.start()
        self.threadLoad.any_signal.connect(self.getThreadResults)
    
    def loadLoadingScreen(self):
        self.toolBar.setVisible(False)
        self.frame.setVisible(False)
        self.image.setVisible(False)
        self.imageName.setVisible(False)
        self.index.setVisible(False)
        self.verticalSlider.setVisible(False)

        self.loadingLabel.setText("Preprocessing scan")
        self.loadingLabel.adjustSize()
        movie = QtGui.QMovie('assets\loading.gif')
        self.loading.setMovie(movie)
        movie.start()

    def loadInfos(self):
        img3D = np.zeros((self.im.shape[1],self.im.shape[2],self.im.shape[3],3),dtype=self.im.dtype)
        for z in range(self.im.shape[1]):
            img3D[z] = cv.cvtColor(self.im[0][z],cv.COLOR_GRAY2RGB)
        self.labelImage = img3D
        self.updateImage()
        self.imageName.setText(self.path.split("/")[-1])
        self.imageName.adjustSize()
        self.imageName.setGeometry((self.frame.size().width()-self.imageName.size().width())//2,80,self.imageName.size().width(),self.imageName.size().height())
        

    def getThreadResults(self,data):
        self.im = data

        self.loading.setVisible(False)
        self.loadingLabel.setVisible(False)

        self.loadInfos()

        self.toolBar.setVisible(True)
        self.frame.setVisible(True)
        self.image.setVisible(True)
        self.imageName.setVisible(True)
        self.index.setVisible(True)
        self.verticalSlider.setVisible(True)

        self.actionNext_Slice.triggered.connect(self.slideRight)
        self.actionLeft_Slice.triggered.connect(self.slideLeft)
        self.actionReset.triggered.connect(self.reset)
        self.actionGoBack.triggered.connect(self.goBack)
        self.verticalSlider.setMinimum(0)
        self.verticalSlider.setMaximum(self.labelImage.shape[0]-1)
        self.verticalSlider.valueChanged.connect(self.slide)
        print("end")

    def updateImage(self):
        pixmap = QImage(self.labelImage[self.innitIndex], self.labelImage.shape[2], self.labelImage.shape[1], self.labelImage.shape[2]*3, QImage.Format_RGB888)
        self.image.setPixmap(QPixmap.fromImage(pixmap))
        self.image.setScaledContents(True)
        self.verticalSlider.setValue(self.innitIndex)
        self.index.setText(f'{self.innitIndex:03}/{self.labelImage.shape[0]-1}')
    
    def slide(self,value):
        self.innitIndex = value
        self.updateImage()
    
    def slideRight(self):
        self.innitIndex = min(self.labelImage.shape[0]-1,self.innitIndex+1)
        self.updateImage()
    
    def slideLeft(self):
        self.innitIndex = max(0,self.innitIndex-1)
        self.updateImage()
    
    def reset(self):
        self.innitIndex = 0
        self.updateImage()
    
    def goBack(self):
        widget.removeWidget(self)


# main
app = QApplication(sys.argv)
welcome = WelcomeScreen()
widget = QtWidgets.QStackedWidget()
widget.addWidget(welcome)
widget.setFixedHeight(800)
widget.setFixedWidth(1200)
widget.setWindowTitle("Lung Cancer detector")
widget.setWindowIcon(QtGui.QIcon("assets/logo.png"))
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")
