from PySide6 import QtCore, QtGui

from PySide6.QtWidgets import QApplication, QTabWidget, QGridLayout, \
    QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton,\
    QWidget, QHBoxLayout, QVBoxLayout, QCheckBox, QComboBox, \
    QGroupBox, QMessageBox, QRadioButton, QTableWidget, QStatusBar, QTableWidgetItem, \
    QHeaderView, QInputDialog, QTextEdit, QSplitter


import pyqtgraph as pg
import os
import pandas as pd
from VIBRO_MODEL import model_class
import functools
from datetime import datetime
from config import DefaultParam
import numpy as np

util = DefaultParam()


class plotter(pg.PlotWidget):
    def __init__(self):
        super().__init__()

        self.setBackground("w")

    def plot_data(self, x, y, actual_col):
        if y.any:
            self.curves = []

            for i, col in enumerate(actual_col):
                color = QtGui.QColor(util.default_colors[col])
                self.curves.append(self.plot(x, y[:, i],
                                   pen={'width': 2, 'color': color,
                                        'style': QtCore.Qt.SolidLine},
                    name='Сигнал'))
        else:
            self.clear()

    def update_data(self, new_data, num_col):
        print('Графики пытаются обновиться')


class table(QTableWidget):
    def update_data(self, new_data):
        print('Таблица {} обновлена'.format(str(self)))

    def __init__(self, is_model_table=bool):
        super().__init__()

        self._is_model_table = is_model_table

        if self._is_model_table:

            self.setRowCount(util.init_channel_count)
            self.setColumnCount(util.init_channel_count)

            init_channel_name = [
                'Channel {}'.format(i+1) for i in range(self.columnCount())]

        else:

            self.setRowCount(util.init_channel_count)

            init_channel_name = ['Name channel', 'Type', 'Unit', 'Min', 'Max',
                                 "RMS", "Mean", 'Variance', 'First quartile',
                                 'Median', 'Third quartile']

            self.setColumnCount(len(init_channel_name))

        self.setHorizontalHeaderLabels(init_channel_name)

        self.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        # self.resizeColumnsToContents()

        self.horizontalHeader().sectionDoubleClicked.connect(
            self.change_horizontal_header)

        self.list_header = init_channel_name

    def put_data(self, data: pd.core.frame.DataFrame):

        def fill_table(self, data):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if self._is_model_table and i == data.shape[0]-1:
                        self.setItem(
                            i, j, QTableWidgetItem('...'))
                    else:
                        self.setItem(i, j, QTableWidgetItem(
                            str(data[i, j])))

        if self._is_model_table:

            self.setColumnCount(data.shape[1])
            self.setRowCount(data.shape[0] + 1)
            self.setHorizontalHeaderLabels(data.columns.to_list())
            fill_table(self, data.values)
            self.create_ckeck_box()

        else:
            self.setRowCount(data.shape[1])
            fill_table(self, data.values)

    def get_data(self, selected_indexes):  # !!!

        selected_indexes = self.selectionModel().selectedIndexes()
        for index in selected_indexes:
            row = index.row()
            col = index.column()
            item = self.model.item(row, col)

            if item is not None:

                data = item.text()
                print(f'Содержимое в строке {row}, столбце {col}: {data}')

    def create_ckeck_box(self):
        self.insertRow(0)
        self.checkboxes = []

        for col in range(self.columnCount()):
            checkbox = QCheckBox()

            self.checkboxes.append(checkbox)
            self.setCellWidget(0, col, checkbox)
            checkbox.setChecked(True)

    # Переименование заголовков
    def change_horizontal_header(self, index):

        old_header = self.horizontalHeaderItem(index).text()
        new_header, ok = QInputDialog.getText(self,
                                              'Change header label for column %d' % index,
                                              'Header:',
                                              QLineEdit.Normal,
                                              old_header)
        dialog = QInputDialog()
        dialog.resize(dialog.sizeHint())
        if ok:
            self.horizontalHeaderItem(
                index).setText(new_header)


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

                log_entry = '{} : {},  Elapsed time: {} c\n'\
                    .format(start_time.strftime("%Y-%m-%d %H:%M:%S"),
                            suffix,
                            round(elapsed_time.total_seconds(), util.round_decimal))

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

        self.create_data_widget()
        self.create_data_layout()

        self.create_spect_widget()
        self.create_spect_layout()

        # self.create_pca_widget()
        # self.create_pca_layout()

        # self.create_damage_widget()
        # self.create_damage_layout()
        self.create_log()
        # Создание вкладок
        self.create_tabs()

    def menubar(self):
        menubar = self.menuBar()

        button_action_open = QtGui.QAction(QtGui.QIcon(""), '&Открыть', self)
        # button_action_open.setStatusTip('Открыть файл с данными для обработки и анализа')
        button_action_open.setShortcut('Ctrl+F')
        button_action_open.triggered.connect(self.click_load_data)

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
        self.data_model_table = table(True)
        self.data_param_table = table(False)

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
        self.data_load_button.clicked.connect(self.click_load_data)
        self.data_load_button.setToolTip('Открыть расположение файла')

        self.data_calc_button = QPushButton("Calculate integral parameters")
        self.data_calc_button.clicked.connect(self.click_calc_param)
        self.data_load_button.setToolTip('Рассчитать статистические парамтеры')

        # Создаем графический виджет
        self.data_plot_widget = plotter()
        self.data_spec_widget = plotter()

        # Создаем поля ввода
        self.data_fs_edit = QLineEdit(str(util.sampling_frequency))
        self.data_fs_edit.setMaximumWidth(50)

        self.fs_validator = QtGui.QIntValidator()
        self.fs_validator.setBottom(1)
        self.data_fs_edit.setValidator(self.fs_validator)

    def create_spect_widget(self):

        # Создание лэйблов
        self.spec_type_window_label = QLabel("Тип оконного преобразования:")
        self.spec_width_window_label = QLabel("Ширина окна:")
        self.spec_overlap_window_label = QLabel("Процент перекрытия:")

        # Чек мать его бокс
        self.spec_window_checkbox = QCheckBox(
            'Использовать оконные преобразования')
        self.spec_curve_checkbox = QCheckBox('Сравнить с кривыми вибраций')

        self.spec_window_checkbox.setEnabled(True)
        self.spec_curve_checkbox.setEnabled(True)

        # Привязка к функции обновления статуса доступа
        self.spec_window_checkbox.stateChanged.connect(self.wind_update_enable)
        self.spec_curve_checkbox.stateChanged.connect(self.curve_update_enable)

        # Поля ввода
        self.spec_width_window_edit = QLineEdit('256')

        self.spec_overlap_edit = QLineEdit('50')

        # Комбовомбо
        self.spec_typewindow_combo = QComboBox()
        self.spec_typewindow_combo.addItems(["Прямоугольое окно", "Окно Ханна",
                                             "Окно Хэмминга", 'Окно Блэкмана'])

        self.spec_typewindow_combo.currentIndexChanged.connect(
            self.switch_window_type)
        self.spec_type_window = 'boxcar'  # !!!

        self.spec_typeVibro_combo = QComboBox()
        self.spec_typeVibro_combo.addItems(
            ['Жетская вибрация', 'Стандартная вибрация'])
        self.spec_typeVibro_combo.currentIndexChanged.connect(
            self.switch_VibroType)

        self.spec_calc_combo = QComboBox()
        self.spec_calc_combo.addItems(['Быстрое преобразование Фурье',
                                       'Спектральная плотность мощности',
                                       'Спектрограмма'])

        self.spec_curve_combo = QComboBox()
        self.spec_curve_combo.addItems(['B2', 'B', 'B3', 'C', 'D', 'E'])
        self.spec_curve_combo.currentIndexChanged.connect(self.switch_curve)

        # Graph spec
        self.spec_widget = pg.PlotWidget()
        self.spec_widget.setBackground("w")

        self.spec_plot_curve = self.spec_widget.plot()

        # Graph form
        self.spec_widget = pg.PlotWidget()
        self.spec_widget.setBackground("w")

        self.spec_plot_curve = self.spec_widget.plot()

        # Buttons
        self.spec_calc_button = QPushButton("Отобразить")
        self.spec_calc_button.clicked.connect(self.spec_calc)

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

    def create_spect_layout(self):

        # Глобальный лэйап
        self.spec_lay = QHBoxLayout(self.widget)

        self.spec_option_lay = QVBoxLayout(self.widget)

        # Группа по оконному преобразованию
        self.spec_option_lay.addWidget(self.spec_window_checkbox)

        self.spec_window_group = QGroupBox('Оконное преобразование')

        self.spec_window_groupLay = QGridLayout(self.widget)

        self.spec_window_groupLay.addWidget(self.spec_type_window_label, 0, 0)
        self.spec_window_groupLay.addWidget(self.spec_typewindow_combo, 0, 1)

        self.spec_window_groupLay.addWidget(self.spec_width_window_label, 1, 0)
        self.spec_window_groupLay.addWidget(self.spec_width_window_edit, 1, 1)

        self.spec_window_groupLay.addWidget(
            self.spec_overlap_window_label, 2, 0)
        self.spec_window_groupLay.addWidget(self.spec_overlap_edit, 2, 1)

        self.spec_window_group.setLayout(self.spec_window_groupLay)

        self.spec_option_lay.addWidget(self.spec_window_group)

        self.spec_window_group.setEnabled(False)

        self.spec_option_lay.addWidget(QLabel())

        self.spec_option_lay.addWidget(self.spec_curve_checkbox)

        self.spec_curve_group = QGroupBox(
            'Сравнение с кривыми вибраций (КТ-160G/14G)')

        self.spec_curve_groupLay = QVBoxLayout(self.widget)

        self.spec_curve_groupLay.addWidget(self.spec_curve_combo)
        self.spec_curve_groupLay.addWidget(self.spec_typeVibro_combo)

        self.spec_curve_group.setLayout(self.spec_curve_groupLay)

        self.spec_option_lay.addWidget(self.spec_curve_group)

        # Группа порасчету
        self.spec_calc_group = QGroupBox('Анализ')

        self.spec_calc_groupLay = QHBoxLayout(self.widget)

        self.spec_calc_groupLay.addWidget(self.spec_calc_combo)
        self.spec_calc_groupLay.addWidget(self.spec_calc_button)

        self.spec_calc_group.setLayout(self.spec_calc_groupLay)

        self.spec_option_lay.addWidget(self.spec_calc_group)

        # self.spec_calc_group.setEnabled(False)

        self.spec_lay.addLayout(self.spec_option_lay)

        self.spec_lay.addWidget(self.spec_widget)

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

        self.spec_tab = QWidget()
        self.spec_tab.setLayout(self.spec_lay)
        tab_widget.addTab(self.spec_tab, "Spectral analysis")
        self.spec_tab.setEnabled(False)

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
    def click_load_data(self):

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

            self.data_model_table.put_data(
                data.head(util.channel_head_count))

            self.model = model_class(
                data.values, int(self.data_fs_edit.text())
            )

            self.model.register(self.data_model_table)
            self.model.register(self.data_param_table)
            self.model.register(self.data_plot_widget)
            self.model.register(self.data_spec_widget)

            self.channel_name = data.columns.to_list()  # !!!

            self.data_len_data_label1.setText(str(self.model.data.shape[0]))
            self.data_numb_channel_label1.setText(
                str(self.model.data.shape[1]))

            for col, checkbox in enumerate(self.data_model_table.checkboxes):
                checkbox.stateChanged.connect(
                    lambda state, col=col: self.toggle_column(col))

        # QMessageBox.about(self, "Selected File", "Successfully loaded!")

        self.plot_data_lay()
        self.spec_tab.setEnabled(True)

    @status_bar_update('Выполняется расчет параметров...', 'Расчет параметров')
    def click_calc_param(self):
        # Clear table
        # self.data_param_table.clear()

        self.model.calc_integrate_param()

        df_header = pd.DataFrame(self.channel_name)

        df_type = pd.DataFrame(['Vibration' if 'Vibr' in name else 'Not vibration'
                                for name in self.channel_name])

        df_unit = pd.DataFrame(['g' if 'Vibr' in name else 'Not g'
                                for name in self.channel_name])

        df_min = pd.DataFrame(self.model.data_min)
        df_max = pd.DataFrame(self.model.data_max)
        df_rms = pd.DataFrame(self.model.data_rms)
        df_mean = pd.DataFrame(self.model.data_mean)
        df_var = pd.DataFrame(self.model.data_var)
        df_25 = pd.DataFrame(self.model.data_25_percent)
        df_50 = pd.DataFrame(self.model.data_50_percent)
        df_75 = pd.DataFrame(self.model.data_75_percent)

        param = pd.concat([df_header, df_type, df_unit, df_min, df_max, df_rms,
                           df_mean, df_var, df_25, df_50, df_75], axis=1)

        self.data_param_table.put_data(param)

    def toggle_all_columns(self):

        master_state = self.data_master_checkbox.isChecked()

        for checkbox in self.data_model_table.checkboxes:
            checkbox.setChecked(master_state)

    def toggle_column(self, col):

        if self.data_model_table.checkboxes[col].isChecked():
            self.model.insert_data(col)
            self.plot_data_lay()
        else:
            self.model.delete_data(col)
            self.plot_data_lay()

        self.update_head_color()

    def uodate_model(self):
        pass

    @status_bar_update('Выполняется построение графиков...', 'Построение графиков')
    def plot_data_lay(self):

        self.data_plot_widget.clear()
        self.data_spec_widget.clear()

        self.model.get_fft()
        # ======================================================================
        self.data_plot_widget.setLabel('bottom', 'Time', 's',
                                       title_font=QtGui.QFont("Arial", 14),
                                       units_font=QtGui.QFont("Arial", 12))

        self.data_plot_widget.setLabel('left', 'Signal', 'V',
                                       title_font=QtGui.QFont("Arial", 14),
                                       units_font=QtGui.QFont("Arial", 12))

        self.data_plot_widget.addLegend()
        self.data_plot_widget.showGrid(x=True, y=True, alpha=0.3)

        self.data_plot_widget.plot_data(
            self.model.t, self.model.data, self.model.actual_col)
        # ======================================================================
        self.data_spec_widget.setLabel('bottom', 'Freq', 'Hz',
                                       title_font=QtGui.QFont("Arial", 14),
                                       units_font=QtGui.QFont("Arial", 12))

        self.data_spec_widget.setLabel('left', 'FFT', 'V',
                                       title_font=QtGui.QFont("Arial", 14),
                                       units_font=QtGui.QFont("Arial", 12))

        self.data_spec_widget.addLegend()
        self.data_spec_widget.showGrid(x=True, y=True, alpha=0.3)

        self.data_spec_widget.plot_data(
            self.model.f, self.model.fft, self.model.actual_col)
        # ======================================================================
        self.update_head_color()

    def update_head_color(self):

        for i in range(len(self.channel_name)):
            color = QtGui.QColor(util.default_colors[i])
            self.data_model_table.checkboxes[i].setStyleSheet(
                "background-color: " + 'white')

        for i in self.model.actual_col:
            color = QtGui.QColor(util.default_colors[i])
            self.data_model_table.checkboxes[i].setStyleSheet(
                "background-color: " + color.name())

    def reference_call(self):
        pass

    def clear_all(self):
        pass

    def generate_data(self):

        self.model.generate_data()

    def data_filt(self):
        pass

    def data_smooth(self):
        pass

    def data_quant(self):
        pass

    def print_report(self):
        pass
       # self.data_saveplot_button.setEnabled(True)

    def wind_update_enable(self, state):

        if state == 2:
            self.spec_window_group.setEnabled(True)

        else:

            self.spec_window_group.setEnabled(False)
            self.spec_type_window = "boxcar"

            # self.spec_width_window_edit.setText(self.data_fs_edit.text)

    def curve_update_enable(self, state):

        if state == 2:

            self.spec_curve_group.setEnabled(True)

        else:

            self.spec_curve_group.setEnabled(False)

    def switch_window_type(self, index):

        list_of_types = ["boxcar", "hann", "hamming", 'blackman']

        self.spec_type_window = list_of_types[index]

    def switch_VibroType(self, index):

        list_of_types = [2, 1]

        self.spec_typeVibro = list_of_types[index]

    def switch_curve(self, index):

        curve = {'B2': [(10, 40, 100, 500, 2000),         (0.003, 0.003, 0.0005, 0.0005, 0.00032)],
                 'B': [(10, 40, 100, 500, 2000),         (0.012, 0.012, 0.02,   0.02,   0.00013)],
                 'B3': [(10, 31, 100, 500, 2000),         (0.02,  0.02,  0.02,   0.02,   0.00013)],
                 'C': [(10, 40, 54.7, 500, 2000),         (0.012, 0.012, 0.02,   0.02,   0.00126)],
                 'D': [(10, 28, 40, 250, 500, 2000),      (0.02,  0.02,  0.04,   0.04,   0.08, 0.02)],
                 'E': [(10, 28, 40, 100, 250, 500, 2000), (0.02,  0.02,  0.04,   0.04,   0.08, 0.08, 0.00505)]}

        curve = list(curve.values())

        self.spec_typeCurve = curve[index]

    def spec_calc(self):

        def fft(self):

            fs = int(self.data_fs_edit.text())

            self.model.get_fft()

            # Удаление всех графиков

            self.spec_widget.clear()

            # установка масштаба по оси x и y
            # data_plot_curve.setXRange(0, 10)
            # data_plot_curve.setYRange(-1, 1)

            # Вывод сигнала
            self.spec_widget.setLabel('bottom', 'Time (s)', 's',
                                      title_font=QtGui.QFont("Arial", 14),
                                      units_font=QtGui.QFont("Arial", 12))

            self.spec_widget.setLabel('left', 'Signal', 'V',
                                      title_font=QtGui.QFont("Arial", 14),
                                      units_font=QtGui.QFont("Arial", 12))

            self.spec_widget.addLegend()

            y = self.model.fft

            x = self.model.f

            self.spec_widget.plot(x, y,
                                  pen={'width': 2, 'color': QtGui.QColor(
                                      255, 0, 0, 127), 'style': QtCore.Qt.SolidLine},
                                  name='БПФ')

        def psd(self):

            # Удаление всех графиков

            self.spec_widget.clear()

            fs = int(self.data_fs_edit.text())

            window_width = int(self.spec_width_window_edit.text())
            overlap = int(self.spec_overlap_edit.text())
            window_type = self.spec_type_window

            x = np.array(self.spec_typeCurve[0])
            y = np.array(self.spec_typeCurve[1]) * self.spec_typeVibro

            self.spec_widget.plot(x, y,
                                  pen={'width': 2, 'color': QtGui.QColor(
                                      255, 0, 0, 127), 'style': QtCore.Qt.DashLine},
                                  name='СПМ')

            self.model.get_psd(fs, window_type, window_width, overlap)

            y = self.model.psd
            x = self.model.f

            self.spec_widget.plot(x, y,
                                  pen={'width': 2, 'color': QtGui.QColor(
                                      0, 0, 255, 127), 'style': QtCore.Qt.SolidLine},
                                  name='СПМ')

            # Вывод оригинала, если стоит checkbox
            # if self.data_orig_checkbox.isChecked():

            #     self.data_plot_widget.plot(x = np.arange(0, t , t / len(self.signal_orig.data) ), y = self.signal_orig.data, \
            #         pen = {'width': 2, 'color': QtGui.QColor(0, 255, 0, 127), 'style': QtCore.Qt.SolidLine}, \
            #             name = 'Исходный сигнал')

            #     self.signal_orig.get_psd(fs)

            #     self.data_fft_widget.plot(self.signal_orig.f, self.signal_orig.psd, \
            #         pen = {'width': 2, 'color': QtGui.QColor(255, 0, 255, 127), 'style': QtCore.Qt.SolidLine}, \
            #             name = 'Исходная СПМ')

        def spectogramm(self):
            pass

        if self.spec_calc_combo.currentIndex() == 0:
            fft(self)

        if self.spec_calc_combo.currentIndex() == 1:
            psd(self)

        if self.spec_calc_combo.currentIndex() == 2:
            spectogramm(self)
