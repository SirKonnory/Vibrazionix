from PySide6 import QtCore, QtGui

from PySide6.QtWidgets import QApplication, QTabWidget, QGridLayout, \
    QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton,\
    QWidget, QHBoxLayout, QVBoxLayout, QCheckBox, QComboBox, \
    QGroupBox, QMessageBox, QRadioButton, QTableWidget, QStatusBar, QTableWidgetItem, \
    QHeaderView, QInputDialog, QTextEdit, QSplitter

import pyqtgraph as pg
import os
import pandas as pd
from vibro_model import ModelClass
import functools
from datetime import datetime
import random


class utilit:
    def __init__(self):
        pass


class UI:
    def __init__(self):
        pass


class plotter:
    def __init__(self):
        pass


class MainWindow(QMainWindow):
    def status_bar_update(prefix, suffix):
        def my_decorator(func):
            @functools.wraps(func)
            def wrapped(self, *args, **kwargs):
                self.statusBar.showMessage(prefix)

                start_time = datetime.now()

                result = func(self, *args, **kwargs)

                end_time = datetime.now()

                elapsed_time = (end_time - start_time)

                log_entry = '{} : {},  Elapsed time: {} c\n'.format(start_time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                    suffix,
                                                                    round(elapsed_time.total_seconds(), 2))

                self.log_text_widget.setPlainText(
                    self.log_text_widget.toPlainText() + log_entry)

                self.statusBar.showMessage('Готов к работе')
                return result
            return wrapped
        return my_decorator

    def __init__(self):
        super().__init__()

        self.update_plot = QtCore.Signal

        self.widget = QWidget()
        self.empty_widget = QWidget()
        self.empty_widget.setMinimumWidth(400)

        # Создание менюшки
        self.menubar()
        self.statusBar = QStatusBar()

        # Создание статус-бара
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Готов к работе")

        # Виджеты вкладки инициализации данных
        self.create_data_widget()

        # Компоновка виджетов вкладки инициализации данных
        self.create_data_layout()

        self.create_log()

        # Создание вкладок
        self.create_tabs()

        # self.create_spect_widget()
        # self.create_spect_layout()

        # self.create_pca_widget()
        # self.create_pca_layout()

        # self.create_damage_widget()
        # self.create_damage_layout()

    def menubar(self):
        menubar = self.menuBar()

        button_action_open = QtGui.QAction(QtGui.QIcon(""), '&Открыть', self)
        # button_action_open.setStatusTip('Открыть файл с данными для обработки и анализа')
        button_action_open.setShortcut('Ctrl+F')
        button_action_open.triggered.connect(self.select_file)

        button_action_exit = QtGui.QAction(QtGui.QIcon(""), '&Выйти', self)
        # button_action_exit.setStatusTip('Покинуть "Вибрационикс"')
        button_action_exit.setShortcut('Ctrl+Q')
        button_action_exit.triggered.connect(QApplication.quit)

        button_action_help = QtGui.QAction(QtGui.QIcon(""), '&Помощь', self)
        # button_action_help.setStatusTip('Открыть "ИИЦ 741.033-056-2023"')
        button_action_help.setShortcut('Ctrl+H')
        button_action_help.triggered.connect(self.reference_call)

        button_action_clear = QtGui.QAction(
            QtGui.QIcon(""), '&Очистить все', self)
        button_action_help.triggered.connect(self.clear_all)

        button_action_generate = QtGui.QAction(
            QtGui.QIcon(""), '&Сгенерировать данные', self)
        button_action_help.triggered.connect(self.generate_data)

        button_action_filt = QtGui.QAction(
            QtGui.QIcon(""), '&Фильтрация', self)
        button_action_open.triggered.connect(self.data_filt)
        button_action_smooth = QtGui.QAction(
            QtGui.QIcon(""), '&Сглаживание', self)
        button_action_open.triggered.connect(self.data_filt)
        button_action_quant = QtGui.QAction(
            QtGui.QIcon(""), '&Квантование', self)
        button_action_open.triggered.connect(self.data_filt)

        file_menu = menubar.addMenu('&File')
        file_menu.addAction(button_action_open)
        file_menu.addSeparator()
        file_menu.addAction(button_action_exit)
        file_menu.addSeparator()
        file_menu.addAction(button_action_help)
        file_menu.addSeparator()
        file_menu.addAction(button_action_generate)
        file_menu.addSeparator()
        file_menu.addAction(button_action_clear)

        file_menu = menubar.addMenu('&Tools')
        file_menu.addAction(button_action_filt)
        file_menu.addSeparator()
        file_menu.addAction(button_action_smooth)
        file_menu.addSeparator()
        file_menu.addAction(button_action_quant)
        file_menu.addSeparator()

    def create_data_widget(self):

        # Создаем таблицы
        # ----------------------------------------------------------------------
        self.data_model_table = QTableWidget()

        self.data_model_table.setRowCount(9)
        self.data_model_table.setColumnCount(9)

        # Настроим таблицу
        self.data_model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_model_table.resizeColumnsToContents()
        # self.data_model_table.setFixedSize(700, 200)
        # ----------------------------------------------------------------------

        self.data_param_table = QTableWidget()
        self.data_param = ['Name channel', 'Type', 'Unit', 'Min', 'Max', "RMS", "Mean", 'Variance',
                           'First quartile', 'Median', 'Third quartile']
        self.data_param_table.setRowCount(9)
        self.data_param_table.setColumnCount(len(self.data_param))

        for col, channel in enumerate(self.data_param):
            item = QTableWidgetItem(channel)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.data_param_table.setHorizontalHeaderItem(col, item)

        self.data_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_param_table.resizeColumnsToContents()
        # self.data_model_table.setFixedSize(500, 300)

        # Мастер чек-мать-его-бокс
        self.data_master_checkbox = QCheckBox("Turn all")
        self.data_master_checkbox.setChecked(True)
        self.data_master_checkbox.stateChanged.connect(self.toggle_all_columns)

        # ----------------------------------------------------------------------

        self.data_file_label = QLabel('Open file:')
        self.data_filepath_label = QLabel('')
        self.data_sample_freq_label = QLabel("Sample frequency")
        self.data_numb_channel_label = QLabel("Number of channles:")
        self.data_len_data_label = QLabel("Number of time step:")
        self.data_numb_channel_label1 = QLabel("...")
        self.data_len_data_label1 = QLabel("...")

        # Создаем кнопки
        self.data_load_button = QPushButton("Load data")
        self.data_load_button.clicked.connect(self.select_file)
        # self.data_load_button.setToolTip('Выбери блядский файл')

        self.data_calc_button = QPushButton("Calculate integral parameters")
        self.data_calc_button.clicked.connect(self.calc_integ_param)

        # Создаем графический виджет и его компоненты
        self.data_plot_widget = pg.PlotWidget()
        self.data_plot_widget.setBackground("w")
        self.data_plot_curve = self.data_plot_widget.plot()

        self.data_spec_widget = pg.PlotWidget()
        self.data_spec_widget.setBackground("w")

        # Создаем поля ввода
        self.data_fs_edit = QLineEdit('1024')
        self.data_fs_edit.setMaximumWidth(50)

        self.fs_validator = QtGui.QIntValidator()
        self.fs_validator.setBottom(1)
        self.data_fs_edit.setValidator(self.fs_validator)

    def create_data_layout(self):
        # Глобальный лэйап
        self.data_lay = QGridLayout(self.widget)

        # self.data_lay_model.setSpacing(15)

        # self.data_calc_groupLay.addWidget(self.data_savedata_button, 2, 0, 1, 2)

        self.data_file_group = QGroupBox('Signal initialization')

        self.data_file_groupLay = QHBoxLayout(self.widget)

        self.data_file_groupLay.addWidget(self.data_file_label)
        self.data_file_groupLay.addWidget(self.data_load_button)
        self.data_file_groupLay.addWidget(self.data_filepath_label)

        self.data_file_groupLay.setAlignment(
            self.data_file_label, QtCore.Qt.AlignLeft)
        self.data_file_groupLay.setAlignment(
            self.data_load_button, QtCore.Qt.AlignLeft)

        self.data_file_groupLay.addWidget(self.empty_widget)

        self.data_file_groupLay.addWidget(self.data_fs_edit)
        self.data_file_groupLay.addWidget(self.data_sample_freq_label)

        self.data_file_groupLay.setAlignment(
            self.data_fs_edit, QtCore.Qt.AlignRight)
        self.data_file_groupLay.setAlignment(
            self.data_sample_freq_label, QtCore.Qt.AlignRight)

        self.data_file_group.setLayout(self.data_file_groupLay)

        self.data_lay.addWidget(self.data_file_group, 0, 0, 1, 5)

        self.data_lay.addWidget(self.data_model_table, 2, 0, 12, 5)

        self.data_lay.addWidget(self.data_calc_button, 16, 0, 1, 1)
        self.data_lay.addWidget(self.data_len_data_label, 15, 2, 1, 1)
        self.data_lay.addWidget(self.data_len_data_label1, 15, 3, 1, 1)
        self.data_lay.addWidget(self.data_numb_channel_label, 16, 2, 1, 1)
        self.data_lay.addWidget(self.data_numb_channel_label1, 16, 3, 1, 1)

        self.data_lay.addWidget(self.data_master_checkbox, 15,
                                4, 1, 1, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)

        self.data_lay.addWidget(self.data_param_table, 17, 0, -1, 5)

        self.data_lay.addWidget(self.data_plot_widget, 0, 5, 15, 5)
        self.data_lay.addWidget(self.data_spec_widget, 16, 5, 22, 5)
        # ----------------------------------------------------------------------

    def create_log(self):
        self.log_lay = QVBoxLayout()

        self.log_text_widget = QTextEdit(self)
        self.log_lay.addWidget(self.log_text_widget)

    def create_tabs(self):

        # Создаем объект QTabWidget
        tab_widget = QTabWidget()

        data_tab = QWidget()
        data_tab.setLayout(self.data_lay)
        tab_widget.addTab(data_tab, "Preprocessor")

        # self.pca_tab = QWidget()
        # self.pca_tab.setLayout(self.pca_lay)
        # tab_widget.addTab(self.pca_tab, "Principal component analysis")
        # self.pca_tab.setEnabled(False)

        # self.spec_tab = QWidget()
        # self.spec_tab.setLayout(self.spec_lay)
        # tab_widget.addTab(self.spec_tab, "Spectral analysis")
        # self.spec_tab.setEnabled(False)

        # self.damage_tab = QWidget()

        # self.damage_tab.setLayout(self.damage_lay)
        # tab_widget.addTab(self.damage_tab, "Damage Analysis")
        # self.damage_tab.setEnabled(False)

        log_tab = QWidget()
        log_tab.setLayout(self.log_lay)
        tab_widget.addTab(log_tab, "Log")
        # # Создаем пятую вкладку
        # self.cow_tab = QWidget()

        # image_label = QLabel()

        # cow_lay = QVBoxLayout()

        # pixmap = QtGui.QPixmap("1.png")

        # image_label.setPixmap(pixmap)

        # cow_lay.addWidget(image_label)

        # image_label.setAlignment(QtCore.Qt.AlignCenter)

        # self.cow_tab.setLayout(cow_lay)
        # tab_widget.addTab(self.cow_tab, "Аэродинамика коровы")
        # self.cow_tab.setEnabled(False)

        # Добавляем QTabWidget на главное окно
        self.setCentralWidget(tab_widget)

    @status_bar_update('Выполняется загрузка данных...', 'Загрузка данных')
    def select_file(self):

        f_name, f_type = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Text Document (*.txt)"
                                                                            ";;CSV File(*.csv);;Excel Binary File"
                                                                            "(*.xls);;Excel File(*.xlsx)")
        root, extension = os.path.splitext(f_name)
        extent = extension.casefold()

        self.pathLoadFile = root + extent

        if f_name:
            self.data_filepath_label.setText(f_name)

            if extent == ".xls" or extent == ".xlsx":
                data = pd.read_excel(
                    self.pathLoadFile, index_col=False, dtype=float)
            elif extent == ".txt" or extent == ".csv":
                data = pd.read_csv(self.pathLoadFile, sep='\s+',
                                   index_col=False, dtype=float)

            self.model = ModelClass(
                data.to_numpy(), int(self.data_fs_edit.text()))
            self.channel_name = data.columns.to_list()

            self.data_model_table.setColumnCount(self.model.data.shape[1])
            self.data_model_table.setRowCount(7)

            # Создание чек-мать-их-боксов
            self.checkboxes = []
            for col in range(self.model.data.shape[1]):
                checkbox = QCheckBox()
                self.checkboxes.append(checkbox)
                self.data_model_table.setCellWidget(0, col, checkbox)
                checkbox.setChecked(True)

                # Связываем события флажков в таблице с функцией toggle_column #!!!
                checkbox.stateChanged.connect(
                    lambda state, col=col: self.toggle_column(col))

            # Заоленение таблицы данными
            for i in range(6):
                for j in range(self.model.data.shape[1]):
                    if i == 5:
                        self.data_model_table.setItem(
                            i+1, j, QTableWidgetItem('...'))

                    else:
                        self.data_model_table.setItem(
                            i+1, j, QTableWidgetItem(str(self.model.data[i+1, j])))

            self.data_len_data_label1.setText(str(self.model.data.shape[0]))
            self.data_numb_channel_label1.setText(
                str(self.model.data.shape[1]))

            self.data_model_table.setHorizontalHeaderLabels(self.channel_name)

            self.data_model_table.horizontalHeader().sectionDoubleClicked.connect(
                self.change_horizontal_header)

        # QMessageBox.about(self, "Selected File", "Successfully loaded!")

        # self.buttonSubmitSetup.setEnabled(True)
        # self.buttonClearAll.setEnabled(True)

        self.plot_data()

    @status_bar_update('Выполняется расчет параметров...', 'Расчет параметров')
    # self.data_param = ["Numb of channel",'Type', 'Unit', 'Min', 'Max', "RMS", "Mean", 'STD']
    def calc_integ_param(self):
        # хуево сделано, переписать
        # Clear table
        self.data_param_table.clear()

        # self.data_param_table.setRowCount(0)
        # self.data_param_table.setColumnCount(0)

        # self.data_param_table.setRowCount(len(self.channel_name))
        # self.data_param_table.setColumnCount(len(self.data_param))

        for col, channel in enumerate(self.data_param):
            item = QTableWidgetItem(channel)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.data_param_table.setHorizontalHeaderItem(col, item)

        self.model.calc_integrate_param()

        for param, param_name in enumerate(self.data_param):

            if param_name == 'Name channel':
                for channel, channel_name in enumerate(self.channel_name):
                    self.data_param_table.setItem(
                        channel, param, QTableWidgetItem(channel_name))

            if param_name == 'Type':
                for channel, channel_name in enumerate(self.channel_name):
                    if 'Vibr' in channel_name:
                        self.data_param_table.setItem(
                            channel, param, QTableWidgetItem('Vibration'))
                    else:
                        self.data_param_table.setItem(
                            channel, param, QTableWidgetItem('Not vibration'))

            if param_name == 'Unit':
                for channel, channel_name in enumerate(self.channel_name):
                    if 'Vibr' in channel_name:
                        self.data_param_table.setItem(
                            channel, param, QTableWidgetItem('g'))
                    else:
                        self.data_param_table.setItem(
                            channel, param, QTableWidgetItem('Not g'))

            if param_name == 'Min':
                for channel, channel_name in enumerate(self.channel_name):
                    item = QTableWidgetItem(
                        str(round(self.model.data_min[channel], 3)))
                    self.data_param_table.setItem(channel, param, item)

            if param_name == 'Max':
                for channel, channel_name in enumerate(self.channel_name):
                    item = QTableWidgetItem(
                        str(round(self.model.data_max[channel], 3)))
                    self.data_param_table.setItem(channel, param, item)

            if param_name == 'RMS':
                for channel, channel_name in enumerate(self.channel_name):
                    item = QTableWidgetItem(
                        str(round(self.model.data_rms[channel], 3)))
                    self.data_param_table.setItem(channel, param, item)

            if param_name == 'Mean':
                for channel, channel_name in enumerate(self.channel_name):
                    item = QTableWidgetItem(
                        str(round(self.model.data_mean[channel], 3)))
                    self.data_param_table.setItem(channel, param, item)

            if param_name == 'Variance':
                for channel, channel_name in enumerate(self.channel_name):
                    item = QTableWidgetItem(
                        str(round(self.model.data_var[channel], 3)))
                    self.data_param_table.setItem(channel, param, item)

            if param_name == 'First quartile':
                for channel, channel_name in enumerate(self.channel_name):
                    item = QTableWidgetItem(
                        str(round(self.model.data_25_percent[channel], 3)))
                    self.data_param_table.setItem(channel, param, item)

            if param_name == 'Median':
                for channel, channel_name in enumerate(self.channel_name):
                    item = QTableWidgetItem(
                        str(round(self.model.data_50_percent[channel], 3)))
                    self.data_param_table.setItem(channel, param, item)

            if param_name == 'Third quartile':
                for channel, channel_name in enumerate(self.channel_name):
                    item = QTableWidgetItem(
                        str(round(self.model.data_75_percent[channel], 3)))
                    self.data_param_table.setItem(channel, param, item)

    def toggle_all_columns(self):

        master_state = self.data_master_checkbox.isChecked()

        for checkbox in self.checkboxes:
            checkbox.setChecked(master_state)

    def toggle_column(self, col):

        if self.checkboxes[col].isChecked():
            self.model.insert_data(col)
            self.plot_data()
        else:
            self.model.delete_data(col)
            self.plot_data()

    # Переименование заголовков
    def change_horizontal_header(self, index):

        old_header = self.data_model_table.horizontalHeaderItem(index).text()
        new_header, ok = QInputDialog.getText(self,
                                              'Change header label for column %d' % index,
                                              'Header:',
                                              QLineEdit.Normal,
                                              old_header)
        QInputDialog()
        if ok:
            self.data_model_table.horizontalHeaderItem(
                index).setText(new_header)

    # @status_bar_update('Выполняется построение графиков...','Построение графиков')
    def plot_data(self):
        if self.model.actual_col:
            self.model.get_fft()

            self.data_plot_widget.clear()
            self.data_spec_widget.clear()

            self.data_plot_widget.setLabel('bottom', 'Time', 's',
                                           title_font=QtGui.QFont("Arial", 14),
                                           units_font=QtGui.QFont("Arial", 12))

            self.data_plot_widget.setLabel('left', 'Signal', 'V',
                                           title_font=QtGui.QFont("Arial", 14),
                                           units_font=QtGui.QFont("Arial", 12))

            self.data_plot_widget.addLegend()
            self.data_plot_widget.showGrid(x=True, y=True, alpha=0.3)

            self.data_spec_widget.setLabel('bottom', 'Freq', 'Hz',
                                           title_font=QtGui.QFont("Arial", 14),
                                           units_font=QtGui.QFont("Arial", 12))

            self.data_spec_widget.setLabel('left', 'FFT', 'V',
                                           title_font=QtGui.QFont("Arial", 14),
                                           units_font=QtGui.QFont("Arial", 12))

            self.data_spec_widget.addLegend()
            self.data_spec_widget.showGrid(x=True, y=True, alpha=0.3)

            self.data_curves = []
            self.data_fft_curves = []

            for i in range(self.model.data.shape[1]):

                color = QtGui.QColor(random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255), 127)

                self.data_curves.append(self.data_plot_widget.plot(self.model.t, self.model.data[:, i],
                                                                   pen={
                                                                       'width': 2, 'color': color, 'style': QtCore.Qt.SolidLine},
                                                                   name='Сигнал')
                                        )

                self.data_fft_curves.append(self.data_spec_widget.plot(self.model.f, self.model.fft[:, i],
                                                                       pen={
                                                                           'width': 2, 'color': color, 'style': QtCore.Qt.SolidLine},
                                                                       name='БПФ')
                                            )
                # self.data_model_table.item(1, i).setBackground(QtGui.QBrush(color))

                self.checkboxes[i].setStyleSheet(
                    "background-color: " + color.name())
                # self.data_model_table.setStyleSheet("background-color: #00FF00;")
                # self..setStyleSheet("background-color: #00FF00;")  # Зеленый цвет (RGB)

        else:
            self.data_plot_widget.clear()
            self.data_spec_widget.clear()

    def reference_call(self):
        pass

    def clear_all(self):
        pass

    def generate_data(self):
        pass

    def data_filt(self):
        pass

    def data_smooth(self):
        pass

    def data_quant(self):
        pass
       # self.data_saveplot_button.setEnabled(True)

   # def filt_data(self):
   #     self.signal.filt_signal_data(self.data_filt_lowfreq_edit.text(), self.data_filt_hieghtfreq_edit.text(), self.fs)
   # def reduce_data(self):
   #     self.signal.reduce_signal_data(self.data_reduce_coeff_edit.text())
   # def quantization_data(self):
   #     self.signal.quantization_signal_data(self.data_level_quant_edit.text())
   # def smoothing_data(self):
   #     self.signal.smooth_signal_data(self.data_smoothing_window_edit.text())

# def create_data_layout(self):
#     #Глобальный лэйап
#     self.data_lay = QGridLayout(self.widget)

#     # self.data_lay_model.setSpacing(15)

#     # self.data_calc_groupLay.addWidget(self.data_savedata_button, 2, 0, 1, 2)

#     self.data_file_group = QGroupBox('Signal initialization')

#     self.data_file_groupLay = QHBoxLayout(self.widget)

#     self.data_file_groupLay.addWidget(self.data_file_label)
#     self.data_file_groupLay.addWidget(self.data_load_button)
#     self.data_file_groupLay.addWidget(self.data_filepath_label)

#     self.data_file_groupLay.setAlignment(self.data_file_label, QtCore.Qt.AlignLeft)
#     self.data_file_groupLay.setAlignment(self.data_load_button, QtCore.Qt.AlignLeft)

#     self.data_file_groupLay.addWidget(self.empty_widget)

#     self.data_file_groupLay.addWidget(self.data_fs_edit)
#     self.data_file_groupLay.addWidget(self.data_sample_freq_label)

#     self.data_file_groupLay.setAlignment(self.data_fs_edit, QtCore.Qt.AlignRight)
#     self.data_file_groupLay.setAlignment(self.data_sample_freq_label, QtCore.Qt.AlignRight)

#     self.data_file_group.setLayout(self.data_file_groupLay)

#     self.data_lay.addWidget(self.data_file_group, 0, 0, 1, 5)

#     self.data_lay.addWidget(self.data_model_table, 2, 0 , 12 , 5)

#     self.data_lay.addWidget(self.data_calc_button, 16, 0, 1, 1)
#     self.data_lay.addWidget(self.data_len_data_label, 15, 2, 1, 1)
#     self.data_lay.addWidget(self.data_len_data_label1, 15, 3, 1, 1)
#     self.data_lay.addWidget(self.data_numb_channel_label, 16, 2, 1, 1)
#     self.data_lay.addWidget(self.data_numb_channel_label1, 16, 3, 1, 1)

#     self.data_lay.addWidget(self.data_master_checkbox, 15, 4, 1, 1, QtCore.Qt.AlignTop|QtCore.Qt.AlignRight)

#     self.data_lay.addWidget(self.data_param_table , 17, 0, -1, 5)

#     self.data_lay.addWidget(self.data_plot_widget, 0, 5, 15, 5)
#     self.data_lay.addWidget(self.data_spec_widget, 16, 5, 22, 5)
#     #----------------------------------------------------------------------
#     # #Лэйап настроек
#     # self.data_lay_model =  QVBoxLayout(self.widget )

#     # #Группа по инициализации сигнала
#     # self.data_file_group = QGroupBox('Инициализация сигнала')

#     # self.data_file_groupLay = QHBoxLayout(self.widget)


#     # self.data_file_groupLay.addWidget(self.data_file_label)
#     # self.data_file_groupLay.addWidget(self.data_load_button)
#     # self.data_file_groupLay.addWidget(self.empty_widget)
#     # self.data_file_groupLay.addWidget(self.empty_widget)

#     # self.data_file_groupLay.addWidget(self.data_fs_edit)
#     # self.data_file_groupLay.addWidget(self.data_sample_freq_label)

#     # self.data_file_group.setLayout(self.data_file_groupLay)

#     # self.data_lay_model.addWidget(self.data_file_group)

#     # self.data_lay_model.addWidget(self.data_model_table)
#     # self.data_lay_model.addWidget(self.empty_widget)
#     # self.data_lay_model.addWidget(self.data_param_table)


#     # #Создание области графиков
#     # self.data_lay_graph = QVBoxLayout(self.widget)

#     # #Добавление графиков
#     # self.data_lay_graph.addWidget(self.data_plot_widget)
#     # self.data_lay_graph.addWidget(self.empty_widget)
#     # self.data_lay_graph.addWidget(self.data_spec_widget)

#     # #Без понятия, работает эта поебота или нет
#     # # self.data_lay_option.setAlignment(QtCore.Qt.AlignTop)
#     # # self.data_lay_option.setAlignment(self.data_filt_lowfreq_label, QtCore.Qt.AlignCenter)
#     # # self.data_lay_option.setAlignment(self.data_filt_hightfreq_label, QtCore.Qt.AlignCenter)
#     # #Хуй знает, вроде работает

#     # # self.data_lay_option.setColumnStretch(0, 1)
#     # # self.data_lay_option.setColumnStretch(1, 1)
#     # # self.data_lay_option.setColumnStretch(2, 1)
#     # # self.data_lay_option.setColumnStretch(3, 1)


#     # self.data_lay.addLayout(self.data_lay_model)
#     # self.data_lay.addLayout(self.data_lay_graph)
