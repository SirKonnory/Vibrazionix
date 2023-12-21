from PySide6 import QtCore, QtGui

from PySide6.QtWidgets import QApplication, QTabWidget, QGridLayout, QMainWindow, \
    QFileDialog, QLabel, QLineEdit, QPushButton, QWidget, QHBoxLayout, QVBoxLayout, \
    QCheckBox, QComboBox, QGroupBox, QTableWidget, QStatusBar, QTableWidgetItem, \
    QInputDialog, QTextEdit, QSplitter, QDialog, QMessageBox, QRadioButton

import pyqtgraph as pg
import os
import pandas as pd
from vibro_model import ModelClass
import functools
from datetime import datetime
from config import Config
import numpy as np

config = Config()


class DialogReduce(QDialog):
    def __init__(self):
        super().__init__()

        label_coeff = QLabel('Коэффициент прореживания сигнала')
        self.edit_coeff = QLineEdit('2')

        ok_button = QPushButton("ОК", self)
        ok_button.clicked.connect(self.accept)

        cancel_button = QPushButton("Отмена", self)
        cancel_button.clicked.connect(self.reject)

        lay_dialog = QGridLayout()
        lay_dialog.addWidget(label_coeff, 0, 0)
        lay_dialog.addWidget(self.edit_coeff, 0, 1)

        lay_dialog.addWidget(ok_button, 2, 0)
        lay_dialog.addWidget(cancel_button, 2, 1)

        self.setLayout(lay_dialog)


class DialogQuant(QDialog):
    def __init__(self):
        super().__init__()

        label_levels = QLabel('Число уровней для квантования сигнала')
        self.edit_levels = QLineEdit('8')

        ok_button = QPushButton("ОК", self)
        ok_button.clicked.connect(self.accept)

        cancel_button = QPushButton("Отмена", self)
        cancel_button.clicked.connect(self.reject)

        lay_dialog = QGridLayout()
        lay_dialog.addWidget(label_levels, 0, 0)
        lay_dialog.addWidget(self.edit_levels, 0, 1)

        lay_dialog.addWidget(ok_button, 2, 0)
        lay_dialog.addWidget(cancel_button, 2, 1)

        self.setLayout(lay_dialog)


class DialogSmooth(QDialog):
    def __init__(self):
        super().__init__()

        label_window = QLabel('Ширина окна скользящего реднего')
        self.edit_window = QLineEdit('8')

        ok_button = QPushButton("ОК", self)
        ok_button.clicked.connect(self.accept)

        cancel_button = QPushButton("Отмена", self)
        cancel_button.clicked.connect(self.reject)

        lay_dialog = QGridLayout()
        lay_dialog.addWidget(label_window, 0, 0)
        lay_dialog.addWidget(self.edit_window, 0, 1)

        lay_dialog.addWidget(ok_button, 2, 0)
        lay_dialog.addWidget(cancel_button, 2, 1)

        self.setLayout(lay_dialog)


class DialogFilter(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Параметры фильтрации сигнала')
        plot_widget = DataPlotter()

        label_rank = QLabel('Порядок фильтра')
        label_lowfreq = QLabel('Нижняя частота, Гц')
        label_heigfreq = QLabel('Верхняя частота, Гц')

        self.edit_rank = QLineEdit('6')
        self.edit_lowfreq = QLineEdit('2')
        self.edit_heigfreq = QLineEdit('6')

        ok_button = QPushButton("ОК", self)
        ok_button.clicked.connect(self.accept)

        cancel_button = QPushButton("Отмена", self)
        cancel_button.clicked.connect(self.reject)

        lay_dialog = QGridLayout()
        lay_dialog.addWidget(plot_widget, 0, 0, 5, 5)

        lay_dialog.addWidget(label_rank, 5, 0)
        lay_dialog.addWidget(self.edit_rank, 5, 1)

        lay_dialog.addWidget(label_lowfreq, 6, 0)
        lay_dialog.addWidget(self.edit_lowfreq, 6, 1)

        lay_dialog.addWidget(label_heigfreq, 7, 0)
        lay_dialog.addWidget(self.edit_heigfreq, 7, 1)

        lay_dialog.addWidget(ok_button, 8, 0)
        lay_dialog.addWidget(cancel_button, 8, 1)

        self.setLayout(lay_dialog)


class DialogProcessor(QDialog):
    def __init__(self):
        super().__init__()
        self.generate = False

        self.setWindowTitle('Параметры генерации сигнала')

        label_numb_signal = QLabel('Номер гармоники')
        label_amplitude = QLabel('Амплитуды гармоник')
        label_freq = QLabel('Частоты гармоник, град')
        label_phase = QLabel('Сдвиг по фазе для гармоник')

        label_duration = QLabel('Длительность сигнала, с')
        label_sampling = QLabel('Частота дискретизации сигнала, Гц')

        self.edit_duration = QLineEdit('10')
        self.edit_sampling = QLineEdit('32')

        label_1 = QLabel("Гармоника №1")
        self.edit1_a = QLineEdit('2')
        self.edit1_f = QLineEdit('7')
        self.edit1_p = QLineEdit('0')

        label_2 = QLabel("Гармоника №2")
        self.edit2_a = QLineEdit('5')
        self.edit2_f = QLineEdit('5')
        self.edit2_p = QLineEdit('90')

        label_3 = QLabel("Гармоника №3")
        self.edit3_a = QLineEdit('7')
        self.edit3_f = QLineEdit('1')
        self.edit3_p = QLineEdit('180')

        label_noise = QLabel('Амплитуда шума')
        self.noise = QLineEdit('10')

        label_count = QLabel('Количество сигналов')
        self.edit_count = QLineEdit('6')

        ok_button = QPushButton("ОК", self)
        ok_button.clicked.connect(self.accept)  # Закрываем диалог при нажатии "ОК"

        cancel_button = QPushButton("Отмена", self)
        cancel_button.clicked.connect(self.reject)  # Закрываем диалог при нажатии "Отмена"

        lay_dialog = QGridLayout()

        lay_dialog.addWidget(label_numb_signal, 0, 0)
        lay_dialog.addWidget(label_amplitude, 0, 1)
        lay_dialog.addWidget(label_freq, 0, 2)
        lay_dialog.addWidget(label_phase, 0, 3)

        lay_dialog.addWidget(label_1, 1, 0)
        lay_dialog.addWidget(self.edit1_a, 1, 1)
        lay_dialog.addWidget(self.edit1_f, 1, 2)
        lay_dialog.addWidget(self.edit1_p, 1, 3)

        lay_dialog.addWidget(label_2, 2, 0)
        lay_dialog.addWidget(self.edit2_a, 2, 1)
        lay_dialog.addWidget(self.edit2_f, 2, 2)
        lay_dialog.addWidget(self.edit2_p, 2, 3)

        lay_dialog.addWidget(label_3, 3, 0)
        lay_dialog.addWidget(self.edit3_a, 3, 1)
        lay_dialog.addWidget(self.edit3_f, 3, 2)
        lay_dialog.addWidget(self.edit3_p, 3, 3)

        lay_dialog.addWidget(label_noise, 4, 0)
        lay_dialog.addWidget(self.noise, 4, 1)

        lay_dialog.addWidget(label_duration, 5, 0)
        lay_dialog.addWidget(self.edit_duration, 5, 1)

        lay_dialog.addWidget(label_sampling, 6, 0)
        lay_dialog.addWidget(self.edit_sampling, 6, 1)

        lay_dialog.addWidget(label_count, 7, 0)
        lay_dialog.addWidget(self.edit_count, 7, 1)

        lay_dialog.addWidget(ok_button, 8, 0)
        lay_dialog.addWidget(cancel_button, 8, 1)

        self.setLayout(lay_dialog)

    def accept(self):
        self.generate = True
        super().accept()

    def reject(self):
        self.generate = False
        super().reject()


class DataPlotter(pg.PlotWidget):
    def __init__(self):

        super().__init__()
        self.curves = []
        self.setBackground("w")

    def plot_data(self, x, y, actual_col, name):
        if y.any:

            for i, col in enumerate(actual_col):
                color = QtGui.QColor(config.default_colors[col])
                self.curves.append(self.plot(x, y[:, i],
                                             pen={'width': 2, 'color': color},
                                             name=name[i]))
        else:
            self.clear()

        self.addLegend(loc='best')
        self.showGrid(x=True, y=True, alpha=0.6)
        self.setXRange(0, x[-1])

    def plot_mesh(self, x, y, z):

        img = pg.ImageItem()
        img.setImage(z, autoLevels=True)

        img.setRect(pg.QtCore.QRectF(y[0], x[0], y[-1] - y[0], x[-1] - x[0]))

        self.setLabel('left', 'Time', units='s')
        self.setLabel('bottom', 'Frequency', units='Hz')

        # Добавляем изображение на график
        self.addItem(img)


class Table(QTableWidget):

    def reinit(self, is_model_table):
        self.setRowCount(0)
        self.setColumnCount(0)

        self.checkboxes = []
        self._is_model_table = is_model_table

        if self._is_model_table:

            self.setRowCount(config.init_channel_count)
            self.setColumnCount(config.init_channel_count)

            init_channel_name = [
                'Канал №{}'.format(i + 1) for i in range(self.columnCount())]

        else:

            self.setRowCount(config.init_channel_count)

            init_channel_name = ['Имя канала', 'Тип', 'Единицы измеренеия', 'Мин', 'Макс',
                                 "СКЗ", "Среднее", 'Дисперсия', '0,25-квантиль',
                                 'Медиана', '0,75-квантиль']

            self.setColumnCount(len(init_channel_name))

        self.setHorizontalHeaderLabels(init_channel_name)

        # self.resizeColumnsToContents()

        self.horizontalHeader().sectionDoubleClicked.connect(
            self.change_horizontal_header)

        self.list_header = init_channel_name

    def __init__(self, is_model_table):
        super().__init__()

        self.checkboxes = []
        self._is_model_table = is_model_table

        if self._is_model_table:

            self.setRowCount(config.init_channel_count)
            self.setColumnCount(config.init_channel_count)

            init_channel_name = [
                'Канал №{}'.format(i + 1) for i in range(self.columnCount())]

        else:

            self.setRowCount(config.init_channel_count)

            init_channel_name = ['Имя канала', 'Тип', 'Единицы измеренеия', 'Мин', 'Макс',
                                 "СКЗ", "Среднее", 'Дисперсия', '0,25-квантиль',
                                 'Медиана', '0,75-квантиль']

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
                            str(table_data[i, j]), ))

        if self._is_model_table:

            self.setColumnCount(data.shape[1])
            self.setRowCount(data.shape[0] + 1)
            self.setHorizontalHeaderLabels(data.columns.to_list())
            fill_table(self, data.round(3).values)
            self.create_check_box()

        else:
            self.setRowCount(data.shape[1])

            fill_table(self, data.values)

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
                                              'Изменение заголовка столбца №%d' % index,
                                              'Загоовок:',
                                              QLineEdit.Normal,
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

                log_entry = '{} : {},  Затраченное время: {} c\n' \
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

        self.spec_war_expl = 0
        self.channel_for_pca = None
        self.pca_method_combo = None
        self.pca_width_edit = None
        self.pca_method_combo = None
        self.spec_war_group = None
        self.spec_typewar = None
        self.spec_channel_combo = None
        self.spec_window_checkbox = None
        self.spec_curve_checkbox = None
        self.spec_widget = None
        self.spec_overlap_edit = None
        self.spec_width_window_edit = None
        self.spec_analysis_type = 1
        self.pca_end_edit = None
        self.pca_start_edit = None
        self.pca_signal_widget = None
        self.pca_values_widget = None
        self.pca_pc_widget = None
        self.pca_cov_widget = None
        self.pca_data_combo = None
        self.pca_tab = None
        self._data_genereted = False
        self.model = ModelClass()
        self.generated_data = False
        self.spec_tab = None
        self.data_numb_channel_label1 = None
        self.data_len_data_label1 = None
        self.spec_curve_group = None
        self.spec_typecurve = list(config.curve.values())[0]
        self.spec_typevibro = 1
        self.spec_type_window = 'boxcar'
        self.spec_window_group = None
        self.data_master_checkbox = None
        self.channel_name = None
        self.data_spec_widget = None
        self.data_plot_widget = None
        self.data_param_table = None
        self.data_fs_edit = None
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
            button_action_open.triggered.connect(self.data_click_load)

            button_action_exit = QtGui.QAction(QtGui.QIcon(""), '&Выйти', self)
            # button_action_exit.setStatusTip('Покинуть "Вибрационикс"')
            button_action_exit.triggered.connect(QApplication.quit)

            button_action_help = QtGui.QAction(QtGui.QIcon(""), '&Помощь', self)
            # button_action_help.setStatusTip('Открыть "ИИЦ 741.033-056-2023"')
            button_action_help.setShortcut('Ctrl+H')
            button_action_help.triggered.connect(self.menu_reference)

            button_action_clear = QtGui.QAction(
                QtGui.QIcon(""), '&Очистить все', self)
            button_action_clear.triggered.connect(self.status_bar_update(
                'Выполняется удаление данных...', 'Удаление данных')(self.menu_clear)
                                                  )

            button_action_generate = QtGui.QAction(
                QtGui.QIcon(""), '&Сгенерировать данные', self)
            button_action_generate.triggered.connect(self.menu_generate)

            button_action_filt = QtGui.QAction(
                QtGui.QIcon(""), '&Фильтрация', self)
            button_action_filt.triggered.connect(self.menu_filter)

            button_action_smooth = QtGui.QAction(
                QtGui.QIcon(""), '&Сглаживание', self)
            button_action_smooth.triggered.connect(self.menu_smooth)

            button_action_quant = QtGui.QAction(
                QtGui.QIcon(""), '&Квантование', self)
            button_action_quant.triggered.connect(self.menu_quant)

            button_action_reduce = QtGui.QAction(
                QtGui.QIcon(""), '&Прореживание', self)
            button_action_reduce.triggered.connect(self.menu_reduce)

            file_menu = menubar.addMenu('&Файл')
            file_menu.addAction(button_action_open)
            file_menu.addSeparator()
            file_menu.addAction(button_action_generate)
            file_menu.addSeparator()
            file_menu.addAction(button_action_clear)
            file_menu.addSeparator()
            file_menu.addAction(button_action_help)
            file_menu.addSeparator()
            file_menu.addAction(button_action_exit)

            file_menu = menubar.addMenu('&Инструменты')
            file_menu.addAction(button_action_filt)
            file_menu.addSeparator()
            file_menu.addAction(button_action_smooth)
            file_menu.addSeparator()
            file_menu.addAction(button_action_quant)
            file_menu.addSeparator()
            file_menu.addAction(button_action_reduce)

        def create_data_tab():
            # Создаем таблицы
            self.data_model_table = Table(True)
            self.data_param_table = Table(False)

            # Мастер чек-мать-его-бокс
            self.data_master_checkbox = QCheckBox("Включить/отключить все")
            self.data_master_checkbox.setChecked(True)
            self.data_master_checkbox.stateChanged.connect(self.data_toggle_all)

            data_file_label = QLabel('Открыть файл:')
            data_file_label.setMaximumWidth(100)

            self.data_filepath_label = QLabel('')
            data_sample_freq_label = QLabel("Частота дискретизации:")
            data_numb_channel_label = QLabel("Число каналов:")
            data_len_data_label = QLabel("Число отсчетов:")
            self.data_numb_channel_label1 = QLabel("...")
            self.data_len_data_label1 = QLabel("...")

            # Создаем кнопки
            data_load_button = QPushButton("Загрузить данные")
            data_load_button.setMaximumWidth(200)
            data_load_button.clicked.connect(self.data_click_load)
            data_load_button.setToolTip('Открыть расположение файла')

            data_calc_button = QPushButton("Расчет статистических параметров")
            data_calc_button.setMaximumWidth(200)
            data_calc_button.clicked.connect(self.data_click_calc)
            data_calc_button.setToolTip('Рассчитать статистические параметры')

            # Создаем графический виджет
            self.data_plot_widget = DataPlotter()
            self.data_spec_widget = DataPlotter()

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

            data_file_group = QGroupBox('Инициализация сигнала')

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
            param_lay.addWidget(self.empty_widget)

            param_lay.addWidget(self.data_master_checkbox)

            data_lay.addLayout(param_lay)
            # data_lay.addWidget(self.empty_widget)
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
            spec_type_window_label = QLabel("Тип оконного преобразования:")
            spec_type_window_label.setMinimumWidth(200)
            spec_width_window_label = QLabel("Ширина окна:")
            spec_width_window_label.setMinimumWidth(200)
            spec_overlap_window_label = QLabel("Процент перекрытия:")
            spec_overlap_window_label.setMinimumWidth(200)

            # Чек мать его бокс
            self.spec_window_checkbox = QCheckBox('Использовать оконные преобразования')
            self.spec_curve_checkbox = QCheckBox('Сравнить с кривыми вибраций')
            self.spec_war_checkbox = QCheckBox('Сравнение по зонам самолета')

            self.spec_window_checkbox.setEnabled(True)
            self.spec_curve_checkbox.setEnabled(True)
            self.spec_war_checkbox.setEnabled(True)

            # Привязка к функции обновления статуса доступа
            self.spec_window_checkbox.stateChanged.connect(self.spec_wind_update_enable)
            self.spec_curve_checkbox.stateChanged.connect(self.spec_curve_update_enable)
            self.spec_war_checkbox.stateChanged.connect(self.spec_war_update_enable)

            # Поля ввода
            self.spec_width_window_edit = QLineEdit('256')
            self.spec_overlap_edit = QLineEdit('50')

            # Комбовомбо
            spec_typewindow_combo = QComboBox()
            spec_typewindow_combo.addItems(["Прямоугольое окно", "Окно Ханна",
                                            "Окно Хэмминга", 'Окно Блэкмана'])
            spec_typewindow_combo.currentIndexChanged.connect(self.spec_switch_window_type)
            # self.spec_type_window = 'boxcar'  # !!!

            spec_typevibro_combo = QComboBox()
            spec_typevibro_combo.addItems(['Стандартная вибрация', 'Жетская вибрация'])
            spec_typevibro_combo.currentIndexChanged.connect(self.spec_switch_vibrotype)

            radio_fft = QRadioButton('Спектрограма БПФ')
            radio_psd = QRadioButton('Спектральная плотность мощности')
            radio_psd.setChecked(True)
            radio_spectogramm = QRadioButton('Спектрограмма СПМ')

            radio_psd.toggled.connect(lambda: self.spec_swich_analysis(1))
            radio_fft.toggled.connect(lambda: self.spec_swich_analysis(2))
            radio_spectogramm.toggled.connect(lambda: self.spec_swich_analysis(3))

            # Можно переработать, чтобы было без функций
            spec_curve_combo = QComboBox()
            spec_curve_combo.addItems(['B2', 'B', 'B3', 'C', 'D', 'E'])
            spec_curve_combo.currentIndexChanged.connect(self.spec_switch_curve)

            spec_war_combo = QComboBox()
            spec_war_combo.addItems(['А1', 'А', 'Б', 'В', 'Г', 'Д', 'Ж', 'Е'])
            spec_war_combo.currentIndexChanged.connect(self.spec_switch_zone)

            spec_typewar_combo = QComboBox()
            spec_typewar_combo.addItems(['Полетные режимы', 'Взлет/Посадка'])
            spec_typewar_combo.currentIndexChanged.connect(self.spec_switch_wartype)

            self.spec_channel_combo = QComboBox()
            self.spec_channel_combo.setEnabled(False)

            # Graph spec
            self.spec_widget = DataPlotter()

            # Graph form
            self.spec_widget = DataPlotter()

            # Buttons
            spec_calc_button = QPushButton("Отобразить")
            spec_calc_button.clicked.connect(self.spec_calc)

            # ------------------------------------------------------------------------------------------
            # Create Spec Layout
            # ------------------------------------------------------------------------------------------
            spec_splitter = QSplitter()

            # Глобальный лэйап
            self.spec_lay = QHBoxLayout()

            spec_option_lay = QVBoxLayout()

            # Группа по оконному преобразованию
            spec_option_lay.addWidget(self.spec_window_checkbox)

            self.spec_window_group = QGroupBox('Оконное преобразование')

            self.spec_window_groupLay = QGridLayout()

            self.spec_window_groupLay.addWidget(spec_type_window_label, 0, 0)
            self.spec_window_groupLay.addWidget(spec_typewindow_combo, 0, 1)

            self.spec_window_groupLay.addWidget(spec_width_window_label, 1, 0)
            self.spec_window_groupLay.addWidget(self.spec_width_window_edit, 1, 1)

            self.spec_window_groupLay.addWidget(
                spec_overlap_window_label, 2, 0)
            self.spec_window_groupLay.addWidget(self.spec_overlap_edit, 2, 1)

            self.spec_window_group.setLayout(self.spec_window_groupLay)
            self.spec_window_group.setEnabled(False)

            spec_option_lay.addWidget(self.spec_window_group)

            # --------------------------------------------
            spec_option_lay.addWidget(self.empty_widget)
            spec_option_lay.addWidget(self.spec_curve_checkbox)
            self.spec_curve_group = QGroupBox('Сравнение с кривыми вибраций (КТ-160G/14G)')
            self.spec_curve_group.setEnabled(False)
            self.spec_curve_groupLay = QVBoxLayout()

            self.spec_curve_groupLay.addWidget(spec_curve_combo)
            self.spec_curve_groupLay.addWidget(spec_typevibro_combo)

            self.spec_curve_group.setLayout(self.spec_curve_groupLay)

            spec_option_lay.addWidget(self.spec_curve_group)

            # --------------------------------------------
            spec_option_lay.addWidget(self.empty_widget)
            spec_option_lay.addWidget(self.spec_war_checkbox)
            self.spec_war_group = QGroupBox('Сравнение с нормами по ОТТ ВВС-86')
            self.spec_war_group.setEnabled(False)

            self.spec_war_groupLay = QVBoxLayout()

            self.spec_war_groupLay.addWidget(spec_war_combo)

            self.spec_war_group.setLayout(self.spec_war_groupLay)

            spec_option_lay.addWidget(self.spec_war_group)
            # --------------------------------------------
            # Группа порасчету
            spec_option_lay.addWidget(self.empty_widget)
            spec_calc_group = QGroupBox('Анализ')

            spec_calc_grouplay = QVBoxLayout()

            spec_calc_grouplay.addWidget(radio_psd)
            spec_calc_grouplay.addWidget(radio_fft)
            # spec_calc_grouplay.addWidget(radio_spectogramm)
            spec_calc_grouplay.addWidget(self.spec_channel_combo)

            spec_calc_grouplay.addWidget(spec_calc_button)

            spec_calc_group.setLayout(spec_calc_grouplay)

            spec_option_lay.addWidget(spec_calc_group)

            # spec_calc_group.setEnabled(False)

            spec_widget_lay = QVBoxLayout()
            spec_widget_lay.addWidget(self.spec_widget)

            spec_splitter.addWidget(QWidget())  # Добавляем пустые виджеты для колонок
            spec_splitter.addWidget(QWidget())

            spec_splitter.widget(0).setLayout(spec_option_lay)
            spec_splitter.widget(1).setLayout(spec_widget_lay)
            spec_splitter.setSizes([10, 1270])

            self.spec_lay.addWidget(spec_splitter)

        def create_pca_tab():
            pca_channel_label = QLabel('Выберете канал')
            self.pca_data_combo = QComboBox()

            pca_processing = QPushButton("Расчитать ГК")
            pca_processing.clicked.connect(self.pca_calc)

            pca_replot_button = QPushButton(
                "Перестроить восстановленный ряд")
            pca_replot_button.clicked.connect(self.pca_replot)

            self.pca_method_combo = QComboBox()
            self.pca_method_combo.addItems(
                ['Подход Теплица', 'Подход с использованием траекторной матрицы'])

            pca_update = QPushButton("Сделать новый сигнал исходным")
            pca_update.clicked.connect(self.pca_updating)

            pca_width_label = QLabel('Длина окна')
            self.pca_width_edit = QLineEdit('30')
            # self.pca_width_edit.setFixedWidth(width)

            pca_count_label = QLabel(
                'Число компонент для восстановленного сигнала')
            # self.pca_count_label.setAlignment(QtCore.Qt.AlignCenter)

            pca_start_label = QLabel('С')
            # self.pca_start_label.setAlignment(QtCore.Qt.AlignRight)
            self.pca_start_edit = QLineEdit()
            # self.pca_start_edit.setFixedWidth(width)

            pca_end_label = QLabel('По')
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
            # Create PCA Layout
            # ------------------------------------------------------------------------------------------

            # Слой вкладки
            self.pca_lay = QHBoxLayout(self.widget)

            # Слой левой части
            pca_left_lay = QVBoxLayout(self.widget)

            # Лэйоут настроек
            pca_option_group = QGroupBox('Настройка параметров')

            pca_option_grouplay = QGridLayout(self.widget)

            pca_option_grouplay.addWidget(pca_channel_label, 0, 0)
            pca_option_grouplay.addWidget(self.pca_data_combo, 0, 1, 1, 2)

            pca_option_grouplay.addWidget(pca_width_label, 1, 0)
            pca_option_grouplay.addWidget(self.pca_width_edit, 1, 1)
            pca_option_grouplay.addWidget(self.pca_method_combo, 1, 2)

            pca_option_grouplay.addWidget(pca_processing, 4, 1, 1, 1)

            pca_option_grouplay.setColumnStretch(0, 1)
            pca_option_grouplay.setColumnStretch(1, 1)
            pca_option_grouplay.setColumnStretch(2, 1)

            pca_option_group.setLayout(pca_option_grouplay)

            # Группа постоработки
            pca_post_group = QGroupBox(' Обработка результатов')

            pca_post_grouplay = QGridLayout(self.widget)

            pca_post_grouplay.addWidget(pca_count_label, 0, 0, 1, 4)

            pca_post_grouplay.addWidget(pca_start_label, 1, 0)
            pca_post_grouplay.addWidget(self.pca_start_edit, 1, 1)
            pca_post_grouplay.addWidget(pca_end_label, 1, 2)
            pca_post_grouplay.addWidget(self.pca_end_edit, 1, 3)
            pca_post_grouplay.addWidget(pca_replot_button, 2, 0, 1, 2)
            pca_post_grouplay.addWidget(pca_update, 2, 2, 1, 2)

            pca_post_grouplay.setColumnStretch(0, 0.2)
            pca_post_grouplay.setColumnStretch(1, 1)
            pca_post_grouplay.setColumnStretch(2, 0.2)
            pca_post_grouplay.setColumnStretch(3, 1)

            pca_post_group.setLayout(pca_post_grouplay)

            pca_left_lay.addWidget(pca_option_group)
            pca_left_lay.addWidget(pca_post_group)
            pca_left_lay.addWidget(self.pca_signal_widget)

            self.pca_lay.addLayout(pca_left_lay)

            pca_data_lay_graph = QVBoxLayout(self.widget)

            pca_data_lay_graph.addWidget(self.pca_cov_widget)
            pca_data_lay_graph.addWidget(self.pca_pc_widget)
            pca_data_lay_graph.addWidget(self.pca_values_widget)

            self.pca_lay.addLayout(pca_data_lay_graph)

        def create_log():
            self.log_lay = QVBoxLayout()

            self.log_text_widget = QTextEdit(self)
            self.log_lay.addWidget(self.log_text_widget)

        def create_tabs():
            # Создаем объект QTabWidget
            tab_widget = QTabWidget()

            data_tab = QWidget()
            data_tab.setLayout(self.data_global_lay)
            tab_widget.addTab(data_tab, "Инициализация данных")

            self.pca_tab = QWidget()
            self.pca_tab.setLayout(self.pca_lay)
            tab_widget.addTab(self.pca_tab, "Метод главных компонент")
            self.pca_tab.setEnabled(False)

            self.spec_tab = QWidget()
            self.spec_tab.setLayout(self.spec_lay)
            tab_widget.addTab(self.spec_tab, "Спектральный анализ")
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
        self.empty_widget.setMinimumWidth(100)

        create_menubar()

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Готов к работе")

        create_data_tab()
        create_spect_tab()
        create_pca_tab()
        create_log()
        create_tabs()

    @status_bar_update('Выполняется мольба о помощи...', 'Просьба о помощи')
    def menu_reference(self):

        msg_box = QMessageBox()
        msg_box.setWindowTitle('Справка')
        msg_box.setText('Руководство пользователя Вы можете найти в ИИЦ 741.033-056-2023')
        msg_box.setDetailedText(
            'Или обратитесь за помощью к моему создателю - Гордону С.В.\nТел. 8(495) 777-21-01, доб. 74-90')

        _ = msg_box.exec()

    def menu_clear(self):

        self.data_model_table.reinit(True)
        self.data_param_table.reinit(False)

        self.data_plot_widget.clear()
        self.data_spec_widget.clear()
        self.spec_widget.clear()
        self.pca_cov_widget.clear()
        self.pca_pc_widget.clear()
        self.pca_signal_widget.clear()
        self.pca_values_widget.clear()

        self.model = ModelClass()

    @status_bar_update('Выполняется генерация данных...', 'Генерация данных')
    def menu_generate(self):

        dialog = DialogProcessor()
        dialog.exec()

        if dialog.generate:
            self.menu_clear()

            a = [float(dialog.edit1_a.text()), float(dialog.edit2_a.text()), float(dialog.edit3_a.text())]
            f = [float(dialog.edit1_f.text()), float(dialog.edit2_f.text()), float(dialog.edit3_f.text())]
            p = [float(dialog.edit1_p.text()), float(dialog.edit2_p.text()), float(dialog.edit3_p.text())]
            noise = float(dialog.noise.text())
            t = float(dialog.edit_duration.text())
            fs = float(dialog.edit_sampling.text())
            n = int(dialog.edit_count.text())

            self.model.generate_data(a, f, p, noise, t, fs, n)

            self._data_genereted = True
            self.data_fs_edit.setText(str(round(fs)))
            self.data_click_load()

        self._data_genereted = False

    @status_bar_update('Выполняется фильтрация данных...', 'Фильтрация данных')
    def menu_filter(self):

        dialog = DialogFilter()

        result = dialog.exec()

        if result:
            low_freq = dialog.edit_lowfreq.text()
            heig_freq = dialog.edit_heigfreq.text()
            rank = int(dialog.edit_rank.text())
            df = int(float(self.data_fs_edit.text()))
            self.model.filt_signal_data(low_freq, heig_freq, rank, df)

            self.data_plot()

    @status_bar_update('Выполняется сглаживание данных...', 'Сглаживание данных')
    def menu_smooth(self):
        dialog = DialogSmooth()

        result = dialog.exec()

        if result:
            window = dialog.edit_window.text()
            self.model.smooth_signal_data(window)

            self.data_plot()

    def menu_quant(self):

        dialog = DialogQuant()

        result = dialog.exec()

        if result:
            levels = dialog.edit_levels.text()
            self.model.quantization_signal_data(levels)

            self.data_plot()

    @status_bar_update('Выполняется прореживание данных...', 'Прореживание данных')
    def menu_reduce(self):

        dialog = DialogReduce()

        result = dialog.exec()

        if result:
            coeff = dialog.edit_coeff.text()
            self.model.reduce_signal_data(coeff)

            self.data_plot()

    @status_bar_update('Выполняется создание отчета...', 'Создание отчета')
    def menu_report(self):
        pass

    @status_bar_update('Выполняется загрузка данных...', 'Загрузка данных')
    def data_click_load(self):
        data = None
        self.data_model_table.reinit(True)
        self.data_param_table.reinit(False)

        if not self._data_genereted:

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

                self.model.load_data(
                    data.values, int(self.data_fs_edit.text())
                )

        else:
            self.data_filepath_label.setText('Данные сгенерированы пользователем')
            data = pd.DataFrame(data=self.model.data.astype(float),
                                columns=[f'Сигнал №{i + 1}' for i in range(len(self.model.data[0]))])

            self.data_model_table.put_data(
                data.head(config.channel_head_count))

        self.channel_name = data.columns.to_list()  # !!!

        self.data_len_data_label1.setText(str(self.model.data.shape[0]))
        self.data_numb_channel_label1.setText(
            str(self.model.data.shape[1]))

        for col, checkbox in enumerate(self.data_model_table.checkboxes):
            checkbox.stateChanged.connect(
                lambda state, col=col: self.data_toggle_one(col))

        self.data_plot()
        self.spec_tab.setEnabled(True)
        self.pca_tab.setEnabled(True)

    @status_bar_update('Выполняется расчет интегральных параметров...', 'Расчет интегральных параметров')
    def data_click_calc(self):
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

    def data_toggle_all(self):

        master_state = self.data_master_checkbox.isChecked()

        for checkbox in self.data_model_table.checkboxes:
            checkbox.setChecked(master_state)

    def data_toggle_one(self, col):

        if self.data_model_table.checkboxes[col].isChecked():
            self.model.insert_data(col)
            self.data_plot()
        else:
            self.model.delete_data(col)
            self.data_plot()

        self.data_update_head_color()

    def data_plot(self):
        self.pca_data_combo.clear()
        self.spec_channel_combo.clear()

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

        self.data_plot_widget.plot_data(
            self.model.t, self.model.data, self.model.actual_col, self.channel_name)
        # ======================================================================
        self.data_spec_widget.setLabel('bottom', 'Freq', 'Hz',
                                       title_font=QtGui.QFont("Arial", 14),
                                       units_font=QtGui.QFont("Arial", 12))

        self.data_spec_widget.setLabel('left', 'FFT', 'V',
                                       title_font=QtGui.QFont("Arial", 14),
                                       units_font=QtGui.QFont("Arial", 12))

        self.data_spec_widget.addLegend(loc='best')
        self.data_spec_widget.showGrid(x=True, y=True, alpha=0.3)

        self.data_spec_widget.plot_data(
            self.model.f, self.model.fft, self.model.actual_col, self.channel_name)
        # ======================================================================
        self.data_update_head_color()

        for i in range(len(self.model.actual_col)):
            self.pca_data_combo.addItem(self.channel_name[i])
            self.spec_channel_combo.addItem(self.channel_name[i])

    def data_update_head_color(self):

        for i in range(self.model.data.shape[1]):
            # color = QtGui.QColor(config.default_colors[i])
            self.data_model_table.checkboxes[i].setStyleSheet(
                "background-color: " + 'white')

        for i in self.model.actual_col:
            color = QtGui.QColor(config.default_colors[i])
            self.data_model_table.checkboxes[i].setStyleSheet(
                "background-color: " + color.name())

    @status_bar_update('Выполняется спектральный анализ...', 'Спектральный анализ')
    def spec_calc(self):

        def psd():

            self.spec_widget.clear()

            fs = int(self.data_fs_edit.text())

            if self.spec_curve_checkbox.isChecked():
                x = np.array(self.spec_typecurve[0])
                y = np.array(self.spec_typecurve[1]) * self.spec_typevibro

                self.spec_widget.plot(x, y,
                                      pen={'width': 2, 'color': QtGui.QColor(
                                          255, 0, 0, 127), 'style': QtCore.Qt.DashLine})

            if self.spec_window_checkbox.isChecked():

                window_width = int(self.spec_width_window_edit.text())
                overlap = int(self.spec_overlap_edit.text())
                window_type = self.spec_type_window

            else:
                window_width = None
                overlap = None
                window_type = 'hann'

            self.model.get_psd(fs, window_type, window_width, overlap)

            y = self.model.psd
            x = self.model.f

            self.spec_widget.plot_data(x, y, self.model.actual_col, self.channel_name)

        def stft():
            self.spec_widget.clear()

            data = self.model.data[:, self.spec_channel_combo.currentIndex()]
            fs = int(self.data_fs_edit.text())

            if self.spec_window_checkbox.isChecked():

                window_width = int(self.spec_width_window_edit.text())
                overlap = int(self.spec_overlap_edit.text())
                window_type = self.spec_type_window

            else:
                window_width = None
                overlap = None
                window_type = 'hann'

            self.model.get_stft(data, fs, window_type, window_width, overlap)
            self.spec_widget.plot_mesh(self.model.tt, self.model.ff, self.model.zz)

        def spectogramm():

            self.spec_widget.clear()

            data = self.model.data[:, self.spec_channel_combo.currentIndex()]
            fs = int(self.data_fs_edit.text())

            if self.spec_window_checkbox.isChecked():

                window_width = int(self.spec_width_window_edit.text())
                overlap = int(self.spec_overlap_edit.text())
                window_type = self.spec_type_window

            else:
                window_width = None
                overlap = None
                window_type = 'hann'

            self.model.get_spectogram(data, fs, window_type, window_width, overlap)
            self.spec_widget.plot_mesh(self.model.tt, self.model.ff, self.model.zz)

        if self.spec_analysis_type == 1:
            psd()

        if self.spec_analysis_type == 2:
            stft()

        if self.spec_analysis_type == 3:
            spectogramm()

    def spec_wind_update_enable(self, state):

        if state == 2:
            self.spec_window_group.setEnabled(True)

        else:

            self.spec_window_group.setEnabled(False)
            self.spec_type_window = "boxcar"

            # self.spec_width_window_edit.setText(self.data_fs_edit.text)

    def spec_curve_update_enable(self, state):

        if state == 2:

            self.spec_curve_group.setEnabled(True)

        else:

            self.spec_curve_group.setEnabled(False)

    def spec_war_update_enable(self, state):

        if state == 2:

            self.spec_war_group.setEnabled(True)

        else:

            self.spec_war_group.setEnabled(False)

    def spec_switch_window_type(self, index):

        list_of_types = ["boxcar", "hann", "hamming", 'blackman']

        self.spec_type_window = list_of_types[index]

    def spec_switch_vibrotype(self, index):

        list_of_types = [1, 2]
        self.spec_typevibro = list_of_types[index]

    def spec_switch_curve(self, index):

        curve = list(config.curve.values())
        self.spec_typecurve = curve[index]

    def spec_switch_zone(self, index):  # !!!

        zone = list(config.zone.values())
        self.spec_typewar = zone[index]

    def spec_switch_wartype(self, index):

        list_of_types = [0, 1]
        self.spec_war_expl = list_of_types[index]

    def spec_swich_analysis(self, vibro_type):

        if vibro_type == 2 or vibro_type == 3:
            self.spec_channel_combo.setEnabled(True)
        else:
            self.spec_channel_combo.setEnabled(False)

        self.spec_analysis_type = vibro_type

    @status_bar_update('Выполняется расчет главных компонент...', 'Расчет главных компонент')
    def pca_calc(self):

        self.channel_for_pca = self.pca_data_combo.currentIndex()
        data = self.model.data[:, self.channel_for_pca]
        window = int(self.pca_width_edit.text())
        start = self.pca_start_edit.text()
        end = self.pca_end_edit.text()
        method = self.pca_method_combo.currentIndex()

        self.model.pca_compute(data, window, start, end, method)

        self.pca_plot()

    def pca_plot(self):

        self.pca_cov_widget.clear()
        self.pca_pc_widget.clear()
        self.pca_values_widget.clear()
        self.pca_signal_widget.clear()

        fs = float(self.data_fs_edit.text())
        t = len(self.model.data) / fs

        # Ковариационная матрица
        img = pg.ImageItem(self.model.cov)
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

        x = np.arange(0, len(self.model.pc))

        for i in range(4):
            y = self.model.pc[:, i]

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

        x = np.arange(0, len(self.model.lamb))
        y = np.real_if_close(self.model.lamb)

        self.pca_values_widget.plot(x, y, pen={'width': 2, 'color': 'Black'})

        # Восстановленный сигнал

        self.pca_signal_widget.addLegend()

        self.pca_signal_widget.setLabel('bottom', 'Time', 's',
                                        title_font=QtGui.QFont("Arial", 14),
                                        units_font=QtGui.QFont("Arial", 12))

        self.pca_signal_widget.setLabel('left', 'Signal', 'V',
                                        title_font=QtGui.QFont("Arial", 14),
                                        units_font=QtGui.QFont("Arial", 12))

        x = np.arange(0, t, t / len(self.model.data))
        y = self.model.data[:, self.channel_for_pca]

        self.pca_signal_widget.plot(x, y,
                                    pen={'width': 2, 'color': QtGui.QColor(
                                        255, 0, 0, 127), 'style': QtCore.Qt.SolidLine},
                                    name='Исходный сигнал')

        y = self.model.recovered

        self.pca_signal_widget.plot(x, y,
                                    pen={'width': 2, 'color': QtGui.QColor(
                                        0, 0, 255, 255), 'style': QtCore.Qt.SolidLine},
                                    name='Новый сигнал')

        self.pca_signal_widget.showGrid(x=True, y=True)

    def pca_updating(self):

        self.model.data[:, self.channel_for_pca] = self.model.recovered
        self.data_plot()

    def pca_replot(self):

        self.model.recovered = np.sum(self.model.rc[:, int(
            self.pca_start_edit.text()): int(self.pca_end_edit.text())], axis=1)
        self.pca_plot()
