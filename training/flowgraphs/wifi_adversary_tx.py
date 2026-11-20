#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Wifi Adversary Tx
# GNU Radio version: 3.10.1.1

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore


def snipfcn_snippet_time_sync(self):
    import time
    # 2-mboard MIMO: master internal clock, slave off the MIMO cable.
    # set_time_now avoids the N-series PPS crash; a common start_time aligns both.
    time.sleep(1.0)

    self.uhd_usrp_sink_0.set_time_now(uhd.time_spec(0.0))

    time.sleep(0.2)

    t_start = self.uhd_usrp_sink_0.get_time_now() + uhd.time_spec(3.0)

    self.uhd_usrp_sink_0.set_start_time(t_start)

    print(f"DEBUG: synchronized start at {t_start.get_real_secs()}s")


def snippets_main_after_init(tb):
    snipfcn_snippet_time_sync(tb)

from gnuradio import qtgui
import os
CAPTURES = os.environ.get('CAPTURES_ROOT', os.path.expanduser('~/captures'))

class wifi_adversary_tx(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Wifi Adversary Tx", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Wifi Adversary Tx")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "wifi_adversary_tx")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.tx_gain = tx_gain = 13.0000
        self.tx_freq = tx_freq = 5290000000.0
        self.samp_rate = samp_rate = 5e6
        self.pert_file = pert_file = f"{CAPTURES}/attack_assets/ch1_e04_s1.bin"
        self.pdu_length = pdu_length = 500
        self.out_buf_size = out_buf_size = 384000
        self.normalized_gain = normalized_gain = 0.9
        self.lo_offset = lo_offset = 0
        self.interval = interval = 300
        self.freq = freq = 5290000000.0
        self.frame_file = frame_file = f"{CAPTURES}/attack_assets/ch0_run3_107.bin"
        self.epsilon = epsilon = 1.0
        self.encoding = encoding = 0
        self.adv_gain_abs = adv_gain_abs = 20
        self.adv_gain = adv_gain = 17.0000

        ##################################################
        # Blocks
        ##################################################
        self._tx_gain_range = Range(0, 1, 0.01, 0.76, 200)
        self._tx_gain_win = RangeWidget(self._tx_gain_range, self.set_tx_gain, "Normalized gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._tx_gain_win)
        self._tx_freq_range = Range(2.4e9, 5.5e9, 1e6, 5290000000.0, 200)
        self._tx_freq_win = RangeWidget(self._tx_freq_range, self.set_tx_freq, "TX frequency", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._tx_freq_win)
        # Create the options list
        self._samp_rate_options = [5000000.0, 10000000.0, 20000000.0]
        # Create the labels list
        self._samp_rate_labels = ['5 MHz', '10 MHz', '20 MHz']
        # Create the combo box
        self._samp_rate_tool_bar = Qt.QToolBar(self)
        self._samp_rate_tool_bar.addWidget(Qt.QLabel("'samp_rate'" + ": "))
        self._samp_rate_combo_box = Qt.QComboBox()
        self._samp_rate_tool_bar.addWidget(self._samp_rate_combo_box)
        for _label in self._samp_rate_labels: self._samp_rate_combo_box.addItem(_label)
        self._samp_rate_callback = lambda i: Qt.QMetaObject.invokeMethod(self._samp_rate_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._samp_rate_options.index(i)))
        self._samp_rate_callback(self.samp_rate)
        self._samp_rate_combo_box.currentIndexChanged.connect(
            lambda i: self.set_samp_rate(self._samp_rate_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._samp_rate_tool_bar)
        # Create the options list
        self._lo_offset_options = [0, 6000000.0, 11000000.0]
        # Create the labels list
        self._lo_offset_labels = ['0', '6000000.0', '11000000.0']
        # Create the combo box
        self._lo_offset_tool_bar = Qt.QToolBar(self)
        self._lo_offset_tool_bar.addWidget(Qt.QLabel("'lo_offset'" + ": "))
        self._lo_offset_combo_box = Qt.QComboBox()
        self._lo_offset_tool_bar.addWidget(self._lo_offset_combo_box)
        for _label in self._lo_offset_labels: self._lo_offset_combo_box.addItem(_label)
        self._lo_offset_callback = lambda i: Qt.QMetaObject.invokeMethod(self._lo_offset_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._lo_offset_options.index(i)))
        self._lo_offset_callback(self.lo_offset)
        self._lo_offset_combo_box.currentIndexChanged.connect(
            lambda i: self.set_lo_offset(self._lo_offset_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._lo_offset_tool_bar)
        self._epsilon_range = Range(0, 4, 0.01, 3.1, 200)
        self._epsilon_win = RangeWidget(self._epsilon_range, self.set_epsilon, "Perturbation strength (epsilon)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._epsilon_win)
        self._adv_gain_range = Range(0, 1, 0.01, tx_gain, 200)
        self._adv_gain_win = RangeWidget(self._adv_gain_range, self.set_adv_gain, "Adversary TX gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._adv_gain_win)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(('addr0=192.168.10.3,addr1=192.168.10.4', "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,2)),
            ),
            '',
        )
        self.uhd_usrp_sink_0.set_clock_source('internal', 0)
        self.uhd_usrp_sink_0.set_clock_source('mimo', 1)
        self.uhd_usrp_sink_0.set_time_source('mimo', 1)
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(tx_freq, rf_freq = tx_freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_sink_0.set_antenna('J1', 0)   # victim enrolled on J1 (enroll_d5_runs.sh); J2 here made the replay read device_2
        self.uhd_usrp_sink_0.set_gain(tx_gain, 0)

        self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(tx_freq, rf_freq = tx_freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 1)
        self.uhd_usrp_sink_0.set_antenna('J1', 1)   # adversary on J1: +7.9 dB vs J2 (measured 8/13)
        self.uhd_usrp_sink_0.set_gain(adv_gain, 1)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.qtgui_freq_sink_pert = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "Adversary Perturbation Spectrum", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_pert.set_update_time(0.10)
        self.qtgui_freq_sink_pert.set_y_axis(-140, 10)
        self.qtgui_freq_sink_pert.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_pert.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_pert.enable_autoscale(False)
        self.qtgui_freq_sink_pert.enable_grid(False)
        self.qtgui_freq_sink_pert.set_fft_average(1.0)
        self.qtgui_freq_sink_pert.enable_axis_labels(True)
        self.qtgui_freq_sink_pert.enable_control_panel(False)
        self.qtgui_freq_sink_pert.set_fft_window_normalized(False)



        labels = ['Perturbation', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["magenta", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_pert.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_pert.set_line_label(i, labels[i])
            self.qtgui_freq_sink_pert.set_line_width(i, widths[i])
            self.qtgui_freq_sink_pert.set_line_color(i, colors[i])
            self.qtgui_freq_sink_pert.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_pert_win = sip.wrapinstance(self.qtgui_freq_sink_pert.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_pert_win)
        self._pdu_length_range = Range(0, 1500, 1, 500, 200)
        self._pdu_length_win = RangeWidget(self._pdu_length_range, self.set_pdu_length, "'pdu_length'", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._pdu_length_win)
        self._normalized_gain_range = Range(0, 1, 0.01, 0.9, 200)
        self._normalized_gain_win = RangeWidget(self._normalized_gain_range, self.set_normalized_gain, "'normalized_gain'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._normalized_gain_win)
        self._interval_range = Range(10, 1000, 1, 300, 200)
        self._interval_win = RangeWidget(self._interval_range, self.set_interval, "'interval'", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._interval_win)
        # Create the options list
        self._freq_options = [2412000000.0, 2417000000.0, 2422000000.0, 2427000000.0, 2432000000.0, 2437000000.0, 2442000000.0, 2447000000.0, 2450000000.0, 2457000000.0, 2462000000.0, 2467000000.0, 2472000000.0, 2484000000.0, 5170000000.0, 5180000000.0, 5190000000.0, 5200000000.0, 5210000000.0, 5220000000.0, 5230000000.0, 5240000000.0, 5250000000.0, 5260000000.0, 5270000000.0, 5280000000.0, 5290000000.0, 5300000000.0, 5310000000.0, 5320000000.0, 5500000000.0, 5510000000.0, 5520000000.0, 5530000000.0, 5540000000.0, 5550000000.0, 5560000000.0, 5570000000.0, 5580000000.0, 5590000000.0, 5600000000.0, 5610000000.0, 5620000000.0, 5630000000.0, 5640000000.0, 5660000000.0, 5670000000.0, 5680000000.0, 5690000000.0, 5700000000.0, 5710000000.0, 5720000000.0, 5745000000.0, 5755000000.0, 5765000000.0, 5775000000.0, 5785000000.0, 5795000000.0, 5805000000.0, 5825000000.0, 5860000000.0, 5870000000.0, 5880000000.0, 5890000000.0, 5900000000.0, 5910000000.0, 5920000000.0]
        # Create the labels list
        self._freq_labels = ['  1 | 2412.0 | 11g', '  2 | 2417.0 | 11g', '  3 | 2422.0 | 11g', '  4 | 2427.0 | 11g', '  5 | 2432.0 | 11g', '  6 | 2437.0 | 11g', '  7 | 2442.0 | 11g', '  8 | 2447.0 | 11g', '  9 | 2452.0 | 11g', ' 10 | 2457.0 | 11g', ' 11 | 2462.0 | 11g', ' 12 | 2467.0 | 11g', ' 13 | 2472.0 | 11g', ' 14 | 2484.0 | 11g', ' 34 | 5170.0 | 11a', ' 36 | 5180.0 | 11a', ' 38 | 5190.0 | 11a', ' 40 | 5200.0 | 11a', ' 42 | 5210.0 | 11a', ' 44 | 5220.0 | 11a', ' 46 | 5230.0 | 11a', ' 48 | 5240.0 | 11a', ' 50 | 5250.0 | 11a', ' 52 | 5260.0 | 11a', ' 54 | 5270.0 | 11a', ' 56 | 5280.0 | 11a', ' 58 | 5290.0 | 11a', ' 60 | 5300.0 | 11a', ' 62 | 5310.0 | 11a', ' 64 | 5320.0 | 11a', '100 | 5500.0 | 11a', '102 | 5510.0 | 11a', '104 | 5520.0 | 11a', '106 | 5530.0 | 11a', '108 | 5540.0 | 11a', '110 | 5550.0 | 11a', '112 | 5560.0 | 11a', '114 | 5570.0 | 11a', '116 | 5580.0 | 11a', '118 | 5590.0 | 11a', '120 | 5600.0 | 11a', '122 | 5610.0 | 11a', '124 | 5620.0 | 11a', '126 | 5630.0 | 11a', '128 | 5640.0 | 11a', '132 | 5660.0 | 11a', '134 | 5670.0 | 11a', '136 | 5680.0 | 11a', '138 | 5690.0 | 11a', '140 | 5700.0 | 11a', '142 | 5710.0 | 11a', '144 | 5720.0 | 11a', '149 | 5745.0 | 11a (SRD)', '151 | 5755.0 | 11a (SRD)', '153 | 5765.0 | 11a (SRD)', '155 | 5775.0 | 11a (SRD)', '157 | 5785.0 | 11a (SRD)', '159 | 5795.0 | 11a (SRD)', '161 | 5805.0 | 11a (SRD)', '165 | 5825.0 | 11a (SRD)', '172 | 5860.0 | 11p', '174 | 5870.0 | 11p', '176 | 5880.0 | 11p', '178 | 5890.0 | 11p', '180 | 5900.0 | 11p', '182 | 5910.0 | 11p', '184 | 5920.0 | 11p']
        # Create the combo box
        self._freq_tool_bar = Qt.QToolBar(self)
        self._freq_tool_bar.addWidget(Qt.QLabel("'freq'" + ": "))
        self._freq_combo_box = Qt.QComboBox()
        self._freq_tool_bar.addWidget(self._freq_combo_box)
        for _label in self._freq_labels: self._freq_combo_box.addItem(_label)
        self._freq_callback = lambda i: Qt.QMetaObject.invokeMethod(self._freq_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._freq_options.index(i)))
        self._freq_callback(self.freq)
        self._freq_combo_box.currentIndexChanged.connect(
            lambda i: self.set_freq(self._freq_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._freq_tool_bar)
        # Create the options list
        self._encoding_options = [0, 1, 2, 3, 4, 5, 6, 7]
        # Create the labels list
        self._encoding_labels = ['BPSK 1/2', 'BPSK 3/4', 'QPSK 1/2', 'QPSK 3/4', '16QAM 1/2', '16QAM 3/4', '64QAM 2/3', '64QAM 3/4']
        # Create the combo box
        # Create the radio buttons
        self._encoding_group_box = Qt.QGroupBox("'encoding'" + ": ")
        self._encoding_box = Qt.QHBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._encoding_button_group = variable_chooser_button_group()
        self._encoding_group_box.setLayout(self._encoding_box)
        for i, _label in enumerate(self._encoding_labels):
            radio_button = Qt.QRadioButton(_label)
            self._encoding_box.addWidget(radio_button)
            self._encoding_button_group.addButton(radio_button, i)
        self._encoding_callback = lambda i: Qt.QMetaObject.invokeMethod(self._encoding_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._encoding_options.index(i)))
        self._encoding_callback(self.encoding)
        self._encoding_button_group.buttonClicked[int].connect(
            lambda i: self.set_encoding(self._encoding_options[i]))
        self.top_layout.addWidget(self._encoding_group_box)
        self.blocks_multiply_const_pert = blocks.multiply_const_cc(epsilon)
        self.blocks_file_source_pert = blocks.file_source(gr.sizeof_gr_complex*1, pert_file, True, 0, 0)
        self.blocks_file_source_pert.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_pert.set_min_output_buffer(384000)
        self.blocks_file_source_frame = blocks.file_source(gr.sizeof_gr_complex*1, frame_file, True, 0, 0)
        self.blocks_file_source_frame.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_frame.set_min_output_buffer(384000)
        self._adv_gain_abs_range = Range(0, 31, 5, 20, 200)
        self._adv_gain_abs_win = RangeWidget(self._adv_gain_abs_range, self.set_adv_gain_abs, "Adversary Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._adv_gain_abs_win)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_frame, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.blocks_file_source_frame, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.blocks_file_source_pert, 0), (self.blocks_multiply_const_pert, 0))
        self.connect((self.blocks_multiply_const_pert, 0), (self.qtgui_freq_sink_pert, 0))
        self.connect((self.blocks_multiply_const_pert, 0), (self.uhd_usrp_sink_0, 1))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "wifi_adversary_tx")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.set_adv_gain(self.tx_gain)
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 0)

    def get_tx_freq(self):
        return self.tx_freq

    def set_tx_freq(self, tx_freq):
        self.tx_freq = tx_freq
        self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(self.tx_freq, rf_freq = self.tx_freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(self.tx_freq, rf_freq = self.tx_freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 1)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self._samp_rate_callback(self.samp_rate)
        self.qtgui_freq_sink_pert.set_frequency_range(0, self.samp_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_pert_file(self):
        return self.pert_file

    def set_pert_file(self, pert_file):
        self.pert_file = pert_file
        self.blocks_file_source_pert.open(self.pert_file, True)

    def get_pdu_length(self):
        return self.pdu_length

    def set_pdu_length(self, pdu_length):
        self.pdu_length = pdu_length

    def get_out_buf_size(self):
        return self.out_buf_size

    def set_out_buf_size(self, out_buf_size):
        self.out_buf_size = out_buf_size

    def get_normalized_gain(self):
        return self.normalized_gain

    def set_normalized_gain(self, normalized_gain):
        self.normalized_gain = normalized_gain

    def get_lo_offset(self):
        return self.lo_offset

    def set_lo_offset(self, lo_offset):
        self.lo_offset = lo_offset
        self._lo_offset_callback(self.lo_offset)
        self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(self.tx_freq, rf_freq = self.tx_freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(self.tx_freq, rf_freq = self.tx_freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 1)

    def get_interval(self):
        return self.interval

    def set_interval(self, interval):
        self.interval = interval

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self._freq_callback(self.freq)

    def get_frame_file(self):
        return self.frame_file

    def set_frame_file(self, frame_file):
        self.frame_file = frame_file
        self.blocks_file_source_frame.open(self.frame_file, True)

    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        self.blocks_multiply_const_pert.set_k(self.epsilon)

    def get_encoding(self):
        return self.encoding

    def set_encoding(self, encoding):
        self.encoding = encoding
        self._encoding_callback(self.encoding)

    def get_adv_gain_abs(self):
        return self.adv_gain_abs

    def set_adv_gain_abs(self, adv_gain_abs):
        self.adv_gain_abs = adv_gain_abs

    def get_adv_gain(self):
        return self.adv_gain

    def set_adv_gain(self, adv_gain):
        self.adv_gain = adv_gain
        self.uhd_usrp_sink_0.set_gain(self.adv_gain, 1)




def main(top_block_cls=wifi_adversary_tx, options=None):
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
    snippets_main_after_init(tb)
    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
