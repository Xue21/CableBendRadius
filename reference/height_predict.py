# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'height_predict.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import cv2
import numpy as np
import math
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyrealsense2 as rs


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(788, 719)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(150, 250, 141, 61))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(520, 250, 131, 61))
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(150, 140, 131, 51))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(Form)
        self.lineEdit_2.setGeometry(QtCore.QRect(520, 140, 121, 51))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(200, 110, 111, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(540, 100, 101, 21))
        self.label_2.setObjectName("label_2")

        self.pushButton.clicked.connect(self.show)
        self.pushButton_2.clicked.connect(self.clear)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "预测身高"))
        self.pushButton_2.setText(_translate("Form", "清除"))
        self.label.setText(_translate("Form", "输入身高（cm）"))
        self.label_2.setText(_translate("Form", "预测身高(cm)"))

    def show(self):
        s = self.lineEdit.text() #获取文本框中的值
        self.lineEdit_2.setText(str(s))
    def clear(self):
        self.lineEdit.clear()
        self.lineEdit_2.clear()

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion') #设置窗口风格
    MainWindow = QtWidgets.QMainWindow() #创建窗体对象
    ui = Ui_Form() #创建pyqt设计的窗体对象
    ui.setupUi(MainWindow) #调用pyqt窗体的方法对窗体对象进行初始化设置
    MainWindow.show() #显示窗体
    sys.exit(app.exec_())


