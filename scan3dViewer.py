from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from skimage.measure import marching_cubes

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *  

from PyQt5 import uic
from tools import loadSTL

class ThreadClass(QtCore.QThread):
    any_signal = QtCore.pyqtSignal(object)
    def __init__(self,predImg,parent=None):
        super(ThreadClass,self).__init__(parent)
        self.predImg = predImg

    def run(self):
        print("starting")
        data = loadSTL(self.predImg)
        meshdata = gl.MeshData(vertexes=data[0], faces=data[1])
        mesh = gl.GLMeshItem(meshdata=meshdata, smooth=False, drawFaces=True, drawEdges=True, edgeColor=(0.7, 0.7, 0.7, 1))
        mesh.translate(100.0,0.0,0.0)
        mesh.setColor(QtGui.QColor(200, 200, 200))
        self.any_signal.emit(mesh)

class StlViewer(QMainWindow):
    def __init__(self,predImg):
        super(StlViewer,self).__init__()
        self.setFixedHeight(900)
        self.setFixedWidth(700)

        uic.loadUi("UI files/3dViewer.ui",self)
        self.loadLoadingScreen()
        self.predImg = predImg
        self.mesh = None

        self.threadLoad = ThreadClass(self.predImg,parent=None)
        self.threadLoad.start()
        self.threadLoad.any_signal.connect(self.getThreadResults)
    
    def getThreadResults(self,data):
        self.mesh = data
        self.loading.setVisible(False)
        self.loadingLabel.setVisible(False)
        self.initUI()
        self.viewer.addItem(self.mesh)


    def loadLoadingScreen(self):
        self.loadingLabel.adjustSize()
        movie = QtGui.QMovie('assets\giphy.gif')
        self.loading.setMovie(movie)
        movie.start()

    def initUI(self):
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)

        layout = QVBoxLayout()
        centerWidget.setLayout(layout)

        self.viewer = gl.GLViewWidget()
        layout.addWidget(self.viewer, 1)

        self.viewer.setCameraPosition(distance=100)

        g = gl.GLGridItem()
        g.setSize(1000, 1000)
        g.setSpacing(100, 100)
        self.viewer.addItem(g)
