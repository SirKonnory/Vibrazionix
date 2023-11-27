from PySide6 import QtCore, QtGui

from PySide6.QtWidgets import QApplication, QTabWidget, QGridLayout, QMainWindow, \
    QFileDialog, QLabel, QLineEdit, QPushButton, QWidget, QHBoxLayout, QVBoxLayout, \
    QCheckBox, QComboBox, QGroupBox, QTableWidget, QStatusBar, QTableWidgetItem, \
    QInputDialog, QTextEdit, QSplitter

import pyqtgraph as pg
import os
import pandas as pd
from vibro_model import ModelClass
import functools
from datetime import datetime
from config import Config
import numpy as np

config = Config()


class Plotter(pg.PlotWidget):
    def __init__(self):

        super().__init__()
        self.curves = []
        self.setBackground("w")

    def plot_data(self, x, y, actual_col):
        if y.any:

            for i, col in enumerate(actual_col):
                color = QtGui.QColor(config.default_colors[col])
                self.curves.append(self.plot(x, y[:, i],
                                             pen={'width': 2, 'color': color},
                                             name='Сигнал'))
        else:
            self.clear()

    def update_data(self, new_data, num_col):
        print('Графики пытаются обновиться')


class Table(QTableWidget):
    def update_data(self, new_data):
        print('Таблица {} обновлена'.format(str(self)))

    def __init__(self, is_model_table):
        super().__init__()

        self.checkboxes = []
        self._is_model_table = is_model_table

        if self._is_model_table:

            self.setRowCount(config.init_channel_count)
            self.setColumnCount(config.init_channel_count)

            init_channel_name = [
                'Channel {}'.format(i + 1) for i in range(self.columnCount())]

        else:

            self.setRowCount(config.init_channel_count)

            init_channel_name = ['Name channel', 'Type', 'Unit', 'Min', 'Max',
                                 "RMS", "Mean", 'Variance', 'First quartile',
                                 'Median', 'Third quartile']

            self.setColumnCount(len(init_channel_name))

        self.setHorizontalHeaderLabels(init_channel_name)

        # self.resizeColumnsToContents()

        self.horizontalHeader().sectionDoubleClicked.connect(
            self.change_horizontal_header)

        self.list_header = init_channel_name

    def put_data(self, data):

        def fill_table(table, table_data):

            for i in range(table_data.shape[0]):
                for j in range(table_data.shape[1]):
                    if table._is_model_table and i == table_data.shape[0] - 1:
                        table.setItem(
                            i, j, QTableWidgetItem('...'))
                    else:
                        table.setItem(i, j, QTableWidgetItem(
                            str(table_data[i, j])))

        if self._is_model_table:

            self.setColumnCount(data.shape[1])
            self.setRowCount(data.shape[0] + 1)
            self.setHorizontalHeaderLabels(data.columns.to_list())
            fill_table(self, data.values)
            self.create_check_box()

        else:
            self.setRowCount(data.shape[1])
            fill_table(self, data.values)

    # def get_data(self, selected_indexes):  # !!!
    #
    #     selected_indexes = self.selectionModel().selectedIndexes()
    #     for index in selected_indexes:
    #         row = index.row()
    #         col = index.column()
    #         item = self.model.item(row, col)
    #
    #         if item is not None:
    #
    #             data = item.text()
    #             print(f'Содержимое в строке {row}, столбце {col}: {data}')

    def create_check_box(self):
        self.insertRow(0)

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
                                              old_header)
        dialog = QInputDialog()
        dialog.resize(dialog.sizeHint())
        if ok:
            self.horizontalHeaderItem(
                index).setText(new_header)


class MainWindow(QMainWindow):
    @staticmethod
    def status_bar_update(prefix, suffix):
        def my_decorator(func):
            @functools.wraps(func)
            def wrapped(self, *args, **kwargs):
                self.statusBar.showMessage(prefix)

                start_time = datetime.now()

                result = func(self, *args, **kwargs)

                end_time = datetime.now()

                elapsed_time = (end_time - start_time)

                log_entry = '{} : {},  Elapsed time: {} c\n' \
                    .format(start_time.strftime("%Y-%m-%d %H:%M:%S"),
                            suffix,
                            round(elapsed_time.total_seconds(), config.round_decimal))

                self.log_text_widget.setPlainText(
                    self.log_text_widget.toPlainText() + log_entry)

                self.statusBar.showMessage('Готов к работе')
                return result

            return wrapped

        return my_decorator

    def __init__(self):
        super().__init__()

        self.spec_tab = None
        self.data_numb_channel_label1 = None
        self.data_len_data_label1 = None
        self.spec_curve_group = None
        self.spec_typecurve = None
        self.spec_type_window = None
        self.spec_window_group = None
        self.data_master_checkbox = None
        self.channel_name = None
        self.data_spec_widget = None
        self.data_plot_widget = None
        self.data_param_table = None
        self.data_fs_edit = None
        self.model = None
        self.data_model_table = None
        self.data_filepath_label = None
        self.data_global_lay = None
        self.spec_lay = None
        self.pca_lay = None
        self.log_lay = None

        def create_menubar():
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

        def create_data_tab():
            # Создаем таблицы
            self.data_model_table = Table(True)
            self.data_param_table = Table(False)

            # Мастер чек-мать-его-бокс
            self.data_master_checkbox = QCheckBox("Turn all")
            self.data_master_checkbox.setChecked(True)
            self.data_master_checkbox.stateChanged.connect(self.toggle_all_columns)

            data_file_label = QLabel('Open file:')
            self.data_filepath_label = QLabel('')
            data_sample_freq_label = QLabel("Sample frequency")
            data_numb_channel_label = QLabel("Number of channels:")
            data_len_data_label = QLabel("Number of time step:")
            self.data_numb_channel_label1 = QLabel("...")
            self.data_len_data_label1 = QLabel("...")

            # Создаем кнопки
            data_load_button = QPushButton("Load data")
            data_load_button.clicked.connect(self.click_load_data)
            data_load_button.setToolTip('Открыть расположение файла')

            data_calc_button = QPushButton("Calculate integral parameters")
            data_calc_button.clicked.connect(self.click_calc_param)
            data_load_button.setToolTip('Рассчитать статистические параметры')

            # Создаем графический виджет
            self.data_plot_widget = Plotter()
            self.data_spec_widget = Plotter()

            # Создаем поля ввода
            self.data_fs_edit = QLineEdit(str(config.sampling_frequency))
            self.data_fs_edit.setMaximumWidth(50)

            fs_validator = QtGui.QIntValidator()
            fs_validator.setBottom(1)
            self.data_fs_edit.setValidator(fs_validator)

            # ------------------------------------------------------------------------------------------
            # Create Data Layout
            # ------------------------------------------------------------------------------------------

            data_splitter = QSplitter()

            data_lay = QVBoxLayout()

            data_file_group = QGroupBox('Signal initialization')

            data_file_grouplay = QHBoxLayout()

            data_file_grouplay.addWidget(data_file_label)
            data_file_grouplay.addWidget(data_load_button)
            data_file_grouplay.addWidget(self.data_filepath_label)
            data_file_grouplay.addWidget(self.empty_widget)
            data_file_grouplay.addWidget(self.data_fs_edit)
            data_file_grouplay.addWidget(data_sample_freq_label)

            data_file_group.setLayout(data_file_grouplay)

            data_lay.addWidget(data_file_group)

            data_lay.addWidget(self.data_model_table)

            param_lay = QHBoxLayout()
            param_lay.addWidget(data_len_data_label)
            param_lay.addWidget(self.data_len_data_label1)
            param_lay.addWidget(QWidget())
            param_lay.addWidget(data_numb_channel_label)
            param_lay.addWidget(self.data_numb_channel_label1)
            param_lay.addWidget(QWidget())
            param_lay.addWidget(self.data_master_checkbox)

            data_lay.addLayout(param_lay)

            data_lay.addWidget(data_calc_button)

            data_lay.addWidget(self.data_param_table)

            data_plot_lay = QVBoxLayout()

            data_plot_lay.addWidget(self.data_plot_widget)
            data_plot_lay.addWidget(self.data_spec_widget)

            data_splitter.addWidget(QWidget())  # Добавляем пустые виджеты для колонок
            data_splitter.addWidget(QWidget())

            data_splitter.widget(0).setLayout(data_lay)
            data_splitter.widget(1).setLayout(data_plot_lay)

            self.data_global_lay = QVBoxLayout()
            self.data_global_lay.addWidget(data_splitter)

        def create_spect_tab():

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

            self.spec_typevibro_combo = QComboBox()
            self.spec_typevibro_combo.addItems(
                ['Жетская вибрация', 'Стандартная вибрация'])
            self.spec_typevibro_combo.currentIndexChanged.connect(
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

            # ------------------------------------------------------------------------------------------
            # Create Spec Layout
            # ------------------------------------------------------------------------------------------

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
            self.spec_curve_groupLay.addWidget(self.spec_typevibro_combo)

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

        def create_pca_tab():

            self.pca_processing = QPushButton("Расчитать ГК")
            self.pca_processing.clicked.connect(self.pca_calc)

            self.pca_refresh_button = QPushButton(
                "Перестроить восстановленный ряд")
            self.pca_refresh_button.clicked.connect(self.pca_refresh)

            self.pca_method_combo = QComboBox()
            self.pca_method_combo.addItems(
                ['Подход Теплица', 'Подход с использованием траекторной матрицы'])

            self.pca_update = QPushButton("Сделать новый сигнал исходным")
            self.pca_update.clicked.connect(self.pca_updating)

            self.pca_width_label = QLabel('Длина окна')
            self.pca_width_edit = QLineEdit('30')
            # self.pca_width_edit.setFixedWidth(width)

            self.pca_count_label = QLabel(
                'Число компонент для восстановленного сигнала')
            # self.pca_count_label.setAlignment(QtCore.Qt.AlignCenter)

            self.pca_start_label = QLabel('С')
            # self.pca_start_label.setAlignment(QtCore.Qt.AlignRight)
            self.pca_start_edit = QLineEdit()
            # self.pca_start_edit.setFixedWidth(width)

            self.pca_end_label = QLabel('По')
            # self.pca_end_label.setAlignment(QtCore.Qt.AlignRight)
            self.pca_end_edit = QLineEdit()
            # self.pca_end_edit.setFixedWidth(width)

            self.pca_signal_widget = pg.PlotWidget()
            self.pca_signal_widget.setBackground("w")
            self.pca_signal_plot = self.pca_signal_widget.plot()

            self.pca_pc_widget = pg.PlotWidget()
            self.pca_pc_widget.setBackground("w")
            self.pca_comp_plot = self.pca_pc_widget.plot()

            self.pca_values_widget = pg.PlotWidget()
            self.pca_values_widget.setBackground("w")
            self.pca_values_plot = self.pca_values_widget.plot()

            self.pca_cov_widget = pg.PlotWidget()
            self.pca_cov_widget.setBackground("w")
            self.pca_cov_plot = self.pca_cov_widget.plot()

            # ------------------------------------------------------------------------------------------
            # Create Spec Layout
            # ------------------------------------------------------------------------------------------

            # Слой вкладки
            self.pca_lay = QHBoxLayout(self.widget)

            # Слой левой части
            self.pca_left_lay = QVBoxLayout(self.widget)

            # Лэйоут настроек
            self.pca_option_group = QGroupBox('Настройка параметров')

            self.pca_option_groupLay = QGridLayout(self.widget)

            self.pca_option_groupLay.addWidget(self.pca_width_label, 0, 0)
            self.pca_option_groupLay.addWidget(self.pca_width_edit, 0, 1)
            self.pca_option_groupLay.addWidget(self.pca_method_combo, 0, 2)

            self.pca_option_groupLay.addWidget(self.pca_processing, 3, 0, 1, 3)

            self.pca_option_groupLay.setColumnStretch(0, 1)
            self.pca_option_groupLay.setColumnStretch(1, 1)
            self.pca_option_groupLay.setColumnStretch(2, 1)

            self.pca_option_group.setLayout(self.pca_option_groupLay)

            # Группа постоработки
            self.pca_post_group = QGroupBox(' Обработка результатов')

            self.pca_post_groupLay = QGridLayout(self.widget)

            self.pca_post_groupLay.addWidget(self.pca_count_label, 0, 0, 1, 4)

            self.pca_post_groupLay.addWidget(self.pca_start_label, 1, 0)
            self.pca_post_groupLay.addWidget(self.pca_start_edit, 1, 1)
            self.pca_post_groupLay.addWidget(self.pca_end_label, 1, 2)
            self.pca_post_groupLay.addWidget(self.pca_end_edit, 1, 3)
            self.pca_post_groupLay.addWidget(self.pca_refresh_button, 2, 0, 1, 2)
            self.pca_post_groupLay.addWidget(self.pca_update, 2, 2, 1, 2)

            self.pca_post_groupLay.setColumnStretch(0, 0.2)
            self.pca_post_groupLay.setColumnStretch(1, 1)
            self.pca_post_groupLay.setColumnStretch(2, 0.2)
            self.pca_post_groupLay.setColumnStretch(3, 1)

            self.pca_post_group.setLayout(self.pca_post_groupLay)

            self.pca_left_lay.addWidget(self.pca_option_group)
            self.pca_left_lay.addWidget(self.pca_post_group)
            self.pca_left_lay.addWidget(self.pca_signal_widget)

            self.pca_lay.addLayout(self.pca_left_lay)

            self.pca_data_lay_graph = QVBoxLayout(self.widget)

            self.pca_data_lay_graph.addWidget(self.pca_cov_widget)
            self.pca_data_lay_graph.addWidget(self.pca_pc_widget)
            self.pca_data_lay_graph.addWidget(self.pca_values_widget)

            self.pca_lay.addLayout(self.pca_data_lay_graph)

        def create_log():
            self.log_lay = QVBoxLayout()

            self.log_text_widget = QTextEdit(self)
            self.log_lay.addWidget(self.log_text_widget)

        def create_tabs():
            # Создаем объект QTabWidget
            tab_widget = QTabWidget()

            data_tab = QWidget()
            data_tab.setLayout(self.data_global_lay)
            tab_widget.addTab(data_tab, "Preprocessor")

            self.pca_tab = QWidget()
            self.pca_tab.setLayout(self.pca_lay)
            tab_widget.addTab(self.pca_tab, "Principal component analysis")
            self.pca_tab.setEnabled(False)

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

            self.setCentralWidget(tab_widget)

        self.widget = QWidget()
        self.empty_widget = QWidget()
        self.empty_widget.setMinimumWidth(400)

        create_menubar()

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Готов к работе")

        create_data_tab()
        create_spect_tab()
        create_pca_tab()
        create_log()
        create_tabs()

    @status_bar_update('Выполняется загрузка данных...', 'Загрузка данных')
    def click_load_data(self):
        
        data = None
        
        f_name, f_type = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Text Document (*.txt)"
                                                                            ";;CSV File(*.csv);;Excel Binary File"
                                                                            "(*.xls);;Excel File(*.xlsx)")
        root, extension = os.path.splitext(f_name)
        extent = extension.casefold()

        pathloadfile = root + extent

        if f_name:
            self.data_filepath_label.setText(f_name)

            if extent == ".xls" or extent == ".xlsx":
                data = pd.read_excel(
                    pathloadfile, index_col=False, dtype=float)
            elif extent == ".txt" or extent == ".csv":
                data = pd.read_csv(pathloadfile, sep='\s+',
                                   index_col=False, dtype=float)

            self.data_model_table.put_data(
                data.head(config.channel_head_count))

            self.model = ModelClass(
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
                    functools.partial(self.toggle_column, col))

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
            color = QtGui.QColor(config.default_colors[i])
            self.data_model_table.checkboxes[i].setStyleSheet(
                "background-color: " + 'white')

        for i in self.model.actual_col:
            color = QtGui.QColor(config.default_colors[i])
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

        self.spec_typevibro = list_of_types[index]

    def switch_curve(self, index):

        curve = {'B2': [(10, 40, 100, 500, 2000), (0.003, 0.003, 0.0005, 0.0005, 0.00032)],
                 'B': [(10, 40, 100, 500, 2000), (0.012, 0.012, 0.02, 0.02, 0.00013)],
                 'B3': [(10, 31, 100, 500, 2000), (0.02, 0.02, 0.02, 0.02, 0.00013)],
                 'C': [(10, 40, 54.7, 500, 2000), (0.012, 0.012, 0.02, 0.02, 0.00126)],
                 'D': [(10, 28, 40, 250, 500, 2000), (0.02, 0.02, 0.04, 0.04, 0.08, 0.02)],
                 'E': [(10, 28, 40, 100, 250, 500, 2000), (0.02, 0.02, 0.04, 0.04, 0.08, 0.08, 0.00505)]}

        curve = list(curve.values())

        self.spec_typecurve = curve[index]

    def spec_calc(self):

        def fft(window):

            fs = int(window.data_fs_edit.text())

            window.model.get_fft()

            # Удаление всех графиков

            window.spec_widget.clear()

            # установка масштаба по оси x и y
            # data_plot_curve.setXRange(0, 10)
            # data_plot_curve.setYRange(-1, 1)

            # Вывод сигнала
            window.spec_widget.setLabel('bottom', 'Time (s)', 's',
                                      title_font=QtGui.QFont("Arial", 14),
                                      units_font=QtGui.QFont("Arial", 12))

            window.spec_widget.setLabel('left', 'Signal', 'V',
                                      title_font=QtGui.QFont("Arial", 14),
                                      units_font=QtGui.QFont("Arial", 12))

            window.spec_widget.addLegend()

            y = window.model.fft

            x = window.model.f

            window.spec_widget.plot(x, y,
                                  pen={'width': 2, 'color': QtGui.QColor(
                                      255, 0, 0, 127)},
                                  name='БПФ')

        def psd(window):

            # Удаление всех графиков

            window.spec_widget.clear()

            fs = int(window.data_fs_edit.text())

            window_width = int(window.spec_width_window_edit.text())
            overlap = int(window.spec_overlap_edit.text())
            window_type = window.spec_type_window

            x = np.array(window.spec_typecurve[0])
            y = np.array(window.spec_typecurve[1]) * window.spec_typevibro

            window.spec_widget.plot(x, y,
                                  pen={'width': 2, 'color': QtGui.QColor(
                                      255, 0, 0, 127), 'style': QtCore.Qt.DashLine},
                                  name='СПМ')

            window.model.get_psd(fs, window_type, window_width, overlap)

            y = window.model.psd
            x = window.model.f

            window.spec_widget.plot(x, y,
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

    def pca_calc(self):

        window = int(self.pca_width_edit.text())
        start = self.pca_start_edit.text()
        end = self.pca_end_edit.text()

        method = self.pca_method_combo.currentIndex()

        self.model.pca_compute(window, start, end, method)

        self.pca_plot()


    def pca_plot(self):

        self.pca_cov_widget.clear()
        self.pca_pc_widget.clear()
        self.pca_values_widget.clear()
        self.pca_signal_widget.clear()

        fs = float(self.data_fs_edit.text())
        t = len(self.signal_orig.data) / fs

        # Ковариационная матрица
        img = pg.ImageItem(self.signal.cov)
        self.pca_cov_widget.addItem(img, cmap='thermal', vmin=-1, vmax=1)

        # self.pca_cov_widget.setAspectLocked(True)
        self.pca_cov_widget.setLabel('left', 'Y',
                                     title_font=QtGui.QFont("Arial", 14),
                                     units_font=QtGui.QFont("Arial", 12))

        self.pca_cov_widget.setLabel('bottom', 'X',
                                     title_font=QtGui.QFont("Arial", 14),
                                     units_font=QtGui.QFont("Arial", 12))

        self.pca_cov_widget.showGrid(x=True, y=True)

        # Восстановленные компоненты
        self.pca_pc_widget.setLabel('bottom', 'Samples', 's',
                                    title_font=QtGui.QFont("Arial", 14),
                                    units_font=QtGui.QFont("Arial", 12))

        self.pca_pc_widget.setLabel('left', 'RC', 'V',
                                    title_font=QtGui.QFont("Arial", 14),
                                    units_font=QtGui.QFont("Arial", 12))
        self.pca_pc_widget.addLegend()
        self.pca_pc_widget.showGrid(x=True, y=True)

        x = np.arange(0, len(self.signal.pc))

        for i in range(4):
            y = self.signal.pc[:, i]

            self.pca_pc_widget.plot(x, y,
                                    pen={'width': 2,
                                         'color': QtGui.QColor(np.random.randint(0, 255), np.random.randint(0, 255),
                                                               np.random.randint(0, 255))},
                                    name='RC {}'.format(i))

        # Собственные числа
        self.pca_values_widget.setLabel('bottom', 'Number', 's',
                                        title_font=QtGui.QFont("Arial", 14),
                                        units_font=QtGui.QFont("Arial", 12))

        self.pca_values_widget.setLabel('left', 'Value', 'V',
                                        title_font=QtGui.QFont("Arial", 14),
                                        units_font=QtGui.QFont("Arial", 12))
        self.pca_values_widget.addLegend()
        self.pca_values_widget.showGrid(x=True, y=True)

        x = np.arange(0, len(self.signal.lamb))
        y = np.real_if_close(self.signal.lamb)

        self.pca_values_widget.plot(x, y, pen={'width': 2})

        # Восстановленный сигнал

        self.pca_signal_widget.addLegend()

        self.pca_signal_widget.setLabel('bottom', 'Time', 's',
                                        title_font=QtGui.QFont("Arial", 14),
                                        units_font=QtGui.QFont("Arial", 12))

        self.pca_signal_widget.setLabel('left', 'Signal', 'V',
                                        title_font=QtGui.QFont("Arial", 14),
                                        units_font=QtGui.QFont("Arial", 12))

        x = np.arange(0, t, t / len(self.signal.data))
        y = self.signal.data

        self.pca_signal_widget.plot(x, y,
                                    pen={'width': 2, 'color': QtGui.QColor(
                                        255, 0, 0, 127), 'style': QtCore.Qt.SolidLine},
                                    name='Исходный сигнал')

        y = self.signal.recovered

        self.pca_signal_widget.plot(x, y,
                                    pen={'width': 2, 'color': QtGui.QColor(
                                        0, 0, 255, 255), 'style': QtCore.Qt.SolidLine},
                                    name='Новый сигнал')

    def pca_updating(self):

        self.signal.data = self.signal.recovered

    def pca_refresh(self):

        self.signal.recovered = np.sum(self.signal.rc[:, int(
            self.pca_start_edit.text()): int(self.pca_end_edit.text())], axis=1)
        self.pca_plot()