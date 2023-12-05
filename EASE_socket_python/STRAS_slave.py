import socket
import numpy as np
from struct import unpack, pack
from math import radians
from EASE_socket_python.useful import map_norm2slave_slaveOrder, normalize_q_qp, map_norm2range
from EASE_socket_python.useful import orderRanges, checkInRange
from EASE_socket_python.useful import order_qp2slave, order_slave2qp
from EASE_socket_python.useful import filter_q_dot
# from EASE_socket_python.model_utilities.vtk_render import render_shape
# from EASE_socket_python.model_utilities.camera_model import camera_model
# from model_utilities.robot_piecewise import Robot_piecewise_curvature
# from model_utilities.robot_model import robot_model, robot_twobending
# from model_utilities.stras_model import StrasModel
# import cvxpy as cp
# from struct import pack  # unsed, can replace int().to_bytes(**args)
messageTypeDic = {0: 'CMD',
                  1: 'ACK',
                  2: 'TRACE',
                  3: 'DEBUG',
                  4: 'STATE',
                  41: 'STATEEXTENDED',
                  42: 'AXISINFOS',
                  43: 'SUPERVISIORMESSAGE',
                  44: 'TELEOPINFOSTYPE'}
controllerModeOps = {0: 'INIT', 1: 'ON UNCAL', 2: 'DO CAL',
                     3: 'GO TO INIT POS', 4: 'POS CONTROL',
                     5: 'POS PLANNER(DEBUG)', 6: 'VEL PLANNER(DEBUG)',
                     61: 'EMBEDDEDTELEOP', 62: 'MASTEREXTERNALTELEOP',
                     63: 'SLAVEEXTERNALTELEOP',
                     11: 'ETHERCAT BUS INITIALIZATION', 12: 'BUS DISCONNECTED'}
commandByCMD = {0: 'NONE',
                1: 'CONTROLLER_CMD_ON',
                2: 'CONTROLLER_CMD_OFF',
                3: 'CONTROLLER_CMD_CALIBRATION',
                4: 'CONTROLLER_CMD_GO_TO',  # FOR 1 AXIS AT A TIME
                400: 'CONTROLLER_CMD_REACH_END_POINT',
                5: 'CONTROLLER_CMD_VEL',  # FOR VELOCITY CONTROL
                52: 'CONTROLLER_CMD_IDX_VELOCITY',
                6: 'CONTROLLER_CMD_GO_TO_VEL',  # Debug command
                620: 'CONTROLLER_CMD_IDX_GO_TO_VEL',  # USE THIS COMMAND!
                7: 'CONTROLLER_CMD_GET_STATE',
                8: 'CONTROLLER_CMD_STOP',
                9: 'CMD_DEBUG_FAULT',  # Debug command
                # 10: 'CMD_DEBUG_CALIBRATION_NOSUCCESS',  # Debug command
                # 11: 'CMD_DEBUG_NO_SENSOR_FEEDBACK',  # Debug command
                # 12: 'CMD_DEBUG_ACCELERATION_ERR',  # Debug command
                # 13: 'CMD_DEBUG_TRACKING_ERR',  # Debug command
                # 14: 'CMD_DEBUG_CONTROL_ERR',  # Debug command
                41: 'DEBUG_CMD_SET_POS',  # Debug command
                410: 'DEBUG_CMD_SET_POS_IDX',  # Debug command
                # 50: 'DEBUG_CMD_PUSH_BUTTON',  # Debug command
                # 51: 'DEBUG_CMD_RELEASE_BUTTON',  # Debug command
                # 70: 'DEBUG_CMD_SUPERVISOR_ERROR',  # Debug command
                71: 'CONTROLLER_CMD_SEND_AXIS_INFOS',
                72: 'CONTROLLER_CMD_SEND_STATE_EXTENDED',
                73: 'CONTROLLER_CMD_SUPERIOR_MESSAGE',
                # 79: 'DEBUG_CMD_SUPERVISOR_RESET',  # Debug command
                700: 'CONTROLLER_CMD_GET_CURTIME',
                100: 'CONTROLLER_CMD_RESET'}
axisDict = {0: 'TR', 1: 'RR', 2: 'BR', 3: 'AR',
            4: 'TL', 5: 'RL', 6: 'BL', 7: 'AL',
            8: 'BE1', 9: 'BE2', 10: 'TE', 11: 'RE'}
dofDicRange = {'TR': [0, 60],
               'RR': [-(2.0*np.pi/2.0), (2.0*np.pi/2.0)],
               'BR': [-0.085, 0.085],
               'AR': [0, 1],
               'TL': [0, 60],
               'RL': [-(2.0*np.pi/2.0), (2.0*np.pi/2.0)],
               'BL': [-0.085, 0.085],
               'AL': [0, 1],
               'TE': [-60, 60],
               'RE': [-(2.0*np.pi/2.0), (2.0*np.pi/2.0)],
               'BE1': [-0.085, 0.085],
               'BE2': [-0.085, 0.085]}

qlims_down_joint_pos = {'BR': -np.pi/(2*18.5),  # deg/s
                        'RR': radians(2),  # Rotation in degrees/s
                        'TR': 2,  # Translation in mm/s
                        'BL': -np.pi/(2*18.5),  # deg/s
                        'RL': radians(2),  # Rotation in degrees/s
                        'TL': 2,  # Translation in mm/s
                        'EBH': -np.pi/(2*185),  # HorB-E
                        'EBV': -np.pi/(2*185),  # VertB-E
                        'ER': radians(2),  # Rotation ENDO in degrees/s
                        'ET': 1}  # Translation ENDO in mm/s
qlims_up_joint_pos = {'BR': np.pi/(2*18.5),  # deg/s
                      'RR': radians(360),  # Rotation in degrees/s
                      'TR': 72,  # Translation in mm/s
                      'BL': np.pi/(2*18.5),  # deg/s
                      'RL': radians(360),  # Rotation in degrees/s
                      'TL': 72,  # Translation in mm/s
                      'EBH': np.pi/(2*185),  # HorB-E
                      'EBV': np.pi/(2*185),  # VertB-E
                      'ER': radians(358),  # Rotation ENDO in degrees/s
                      'ET': 100}  # Translation ENDO in mm/s
qlims_down_slave_pos = {'BR': -6.5,  # deg/s
                        'RR': -270,  # Rotation in degrees/s
                        'TR': 2,  # Translation in mm/s
                        'BL': -6.5,  # deg/s
                        'RL': -270,  # Rotation in degrees/s
                        'TL': 2,  # Translation in mm/s
                        'EBH': -9.12,  # HorB-E
                        'EBV': -9.12,  # VertB-E
                        'ER': -90,  # Rotation ENDO in degrees/s
                        'ET': 0.5}  # Translation ENDO in mm/s
qlims_up_slave_pos = {'BR': 6.5,  # deg/s
                      'RR': 360,  # Rotation in degrees/s
                      'TR': 73.5,  # Translation in mm/s
                      'BL': 6.5,  # deg/s
                      'RL': 360,  # Rotation in degrees/s
                      'TL': 73.5,  # Translation in mm/s
                      'EBH': 9.12,  # HorB-E
                      'EBV': 9.12,  # VertB-E
                      'ER': 50,  # Rotation ENDO in degrees/s
                      'ET': 90}  # Translation ENDO in mm/s
qlims_down_master_pos = {'BR': -44,  # deg/s
                         'RR': -163,  # Rotation in degrees/s
                         'TR': 0,  # Translation in mm/s
                         'BL': -44,  # deg/s
                         'RL': -163,  # Rotation in degrees/s
                         'TL': 0,  # Translation in mm/s
                         'EBH': 0,  # HorB-E
                         'EBV': 0,  # VertB-E
                         'ER': 0,  # Rotation ENDO in degrees/s
                         'ET': 0}  # Translation ENDO in mm/s
qlims_up_master_pos = {'BR': 44,  # deg/s
                       'RR': 163,  # Rotation in degrees/s
                       'TR': 105,  # Translation in mm/s
                       'BL': 44,  # deg/s
                       'RL': 163,  # Rotation in degrees/s
                       'TL': 97,  # Translation in mm/s
                       'EBH': 0,  # HorB-E
                       'EBV': 0,  # VertB-E
                       'ER': 0,  # Rotation ENDO in degrees/s
                       'ET': 0}  # Translation ENDO in mm/s

slaveOrder = ['EBV', 'EBH', 'ER', 'ET',
              'TR', 'RR', 'BR',
              'TL', 'RL', 'BL']
masterOrder = ['TL', 'RL', 'BL', 'AL',
               'TR', 'RR', 'BR', 'AR',
               'EBV', 'EBH', 'ET', 'ER']
qpOrder = ['BR', 'RR', 'TR',
           'BL', 'RL', 'TL',
           'EBV', 'EBH', 'ER', 'ET']

lims_down_joint = np.array(
    list(qlims_down_joint_pos.values()), dtype=np.float32)
lims_up_joint = np.array(
    list(qlims_up_joint_pos.values()), dtype=np.float32)
lims_down_master = np.array(
    list(qlims_down_master_pos.values()), dtype=np.float32)
lims_up_master = np.array(
    list(qlims_up_master_pos.values()), dtype=np.float32)
lims_down_slave = np.array(
    list(qlims_down_slave_pos.values()), dtype=np.float32)
lims_up_slave = np.array(
    list(qlims_up_slave_pos.values()), dtype=np.float32)
slaveIndexSocket = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
slaveIndexLabels = ['EBV', 'EBH', 'ER', 'ET',
                    'TR', 'RR', 'BR',
                    'TL', 'RL', 'BL', 'AL']
dicSlaveIndex = dict(zip(slaveIndexLabels, slaveIndexSocket))

# To change in your location
rootPC_global = '/home/jfernandoghe/Documents/EASE_DATASET/fernando/'
rootGITFolder_global = 'Communication/'
configFile = 'datafolders.txt'


class STRASslave:
    def __init__(self, host, port):
        """Initialize a STRAS slave port

        Args:
            host (str): IP address of the STRAS computer
            port (str): Port asigned to the slave
        """
        # Connect
        self.host = host
        self.port = port
        self.STRASslave = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Error handlers
        self.errors = None
        self.errorPairing = False
        self.errorVelIDX = False
        self.errorSendMsg = False
        self.errorRecvMsg = False
        # Axis infos
        self.NumberAxis = 0
        self.ActiveAxis = 0
        self.axisMins = []
        self.axisMaxs = []
        # For internal usage (replace with reference)
        self.cycle = 0
        # Get state
        self.tick = 0
        self.controllerMode = 0
        self.TraceOn = 0
        self.CalibratedRobot = 0
        self.SecurityOnRobot = 0
        self.PowerOnRobot = 0
        self.DcOnRobot = 0
        # Variables from state
        self.axisEnable = []
        self.axisCalibrated = []
        self.axisUsed = []
        self.axisPastVal = []
        self.axisPastValNorm = []
        self.axisCurrVal = np.array([], dtype=np.float64)
        self.axisCurrValVel = np.array([])
        self.axisCurrValNormalized = np.array([], dtype=np.float64)
        self.axisReference = []
        self.axisTorque = []
        self.axisButtons = []
        # For qp output
        self.preliminariesForCompute = False
        self.current_q = np.array([], dtype=np.float32)
        self.past_q = np.array([], dtype=np.float32)
        self.current_q_normalized = np.array([], dtype=np.float32)
        self.optStateType = 'vel'  # by defaul agent control - velocity based
        self.flagInitializeModel = False
        self.modelsComputedFlag = False
        # For reference
        self.axisNames = slaveIndexLabels
        self.axisIdxNames = slaveIndexSocket
        # For visualization purposes
        self.initializeRenderFlag = False
        # For debugging QP
        self.qp_value = 0
        self.Tr_e = 0
        self.Jr_e = 0
        self.Tl_e = 0
        self.Jl_e = 0
        self.Tr_0 = 0
        self.Jr_0 = 0
        self.Tl_0 = 0
        self.Jl_0 = 0
        self.Te = 0
        self.Je = 0

    # --------------------------------------------------------------------------
    # ------------------------ Miscellaneous commands
    # --------------------------------------------------------------------------
    def setTimeOut(self, tx):
        """Sets timeout to the initialized STRAS slave node

        Args:
            tx (float): Number of seconds to wait on the socket
        """
        # ------------------------ No expected message reply
        self.STRASslave.settimeout(float(tx))

    def connect(self, printOpt_=True):
        """Manually connects to the STRAS socket with the previously defined
        configuration

        Args:
            printOpt_ (bool, optional): To print outputs. Defaults to True.
        """
        # ------------------------ No expected message reply
        if self.STRASslave.connect_ex((self.host, self.port)) == 0:
            if printOpt_:
                print('Succesful connection!')
                print('hostname:\t', self.host)
                print('port:\t\t', self.port)
        else:
            self.errorPairing = True

    def closeSocket(self):
        """To close the socket after usage
        """
        # ------------------------ No expected message reply
        self.STRASslave.close()

    def onCMD(self, printOpt_=True):
        """Turn on the CMD capapibility on the STRAS socket, by default turned
        OFF

        Args:
            printOpt_ (bool, optional): To print verbose. Defaults to True.

        Returns:
            bytes : Byte message to be unpacked
        """
        data = self.sendCMDmsg2STRAS(msgtype2send=1,
                                     replyMode=1,
                                     xtraV='',
                                     printOpt=printOpt_)
        self.cycle = self.cycle + 1
        return data

    def offCMD(self, printOpt_=True):
        """Turn Off the CMD capapibility on the STRAS socket, by default turned
        OFF

        Args:
            printOpt_ (bool, optional): To print verbose. Defaults to True.

        Returns:
            bytes : Byte message to be unpacked
        """
        data = self.sendCMDmsg2STRAS(msgtype2send=2,
                                     replyMode=1,
                                     xtraV='',
                                     printOpt=printOpt_)
        self.cycle = self.cycle + 1
        return data

    def stopCMD(self, printOpt_=True):
        """Stop the CMD capapibility on the STRAS socket, by default turned
        OFF. REDUNDANT

        Args:
            printOpt_ (bool, optional): To print verbose. Defaults to True.

        Returns:
            bytes : Byte message to be unpacked
        """
        data = self.sendCMDmsg2STRAS(msgtype2send=8,
                                     replyMode=1,
                                     xtraV='',
                                     printOpt=printOpt_)
        self.cycle = self.cycle + 1
        return data

    def resetCMD(self, printOpt_=True):
        """reset the CMD capapibility on the STRAS socket, by default turned
        OFF. REDUNDANT

        Args:
            printOpt_ (bool, optional): To print verbose. Defaults to True.

        Returns:
            bytes : Byte message to be unpacked
        """
        data = self.sendCMDmsg2STRAS(msgtype2send=1000,
                                     replyMode=1,
                                     xtraV='',
                                     printOpt=printOpt_)
        self.cycle = self.cycle + 1
        return data

    # --------------------------------------------------------------------------
    # ------------------------ SIMULATION
    # --------------------------------------------------------------------------
    def simulatedInit(self, printOpt_=False):
        """Simulate a connectection to the STRAS slave socket

        Args:
            printOpt_ (bool, optional): Verbose. Defaults to False.
        """
        self.NumberAxis = 11
        self.axisMins = [-9.12, -9.12, -90, 0.5,
                         0.5, -270, -6.5,
                         1, -270, -5,
                         -1]
        self.axisMaxs = [9.12, 9.12, 50, 90,
                         73.5, 360, 6.5,
                         73.5, 360, 6.5,
                         9.5]
        self.ActiveAxis = [False]*self.NumberAxis
        if printOpt_:
            for i_ in range(0, self.NumberAxis):
                print(slaveIndexLabels[i_], '', slaveIndexSocket[i_], '\t',
                      ('ACTIVE' if self.ActiveAxis[i_] > 0 else 'INACTIVE'))
                print('\tmin: ', self.axisMins[i_])
                print('\tmax: ', self.axisMaxs[i_])

    def simulatedState(self, printOpt_=False, q_in_=0, q_in_norm=0):
        """Simulate a state update to the STRAS slave socket.

        Args:
            printOpt_ (bool, optional): Verbose. Defaults to False.
            q_in_ (int, optional): List of values to be set on the fake socket;
            In range with STRAS slave Min-Max. Defaults to 0.
            q_in_norm (int, optional): List of  normalized values to be set on
            the fake socket. Defaults to 0.
        """
        if (type(q_in_) == int) and (type(q_in_norm) == int):
            q_simSlave = np.array([0.0067, 0.0069, 0.002, 0.5028,
                                   8.7801, 1.9728, -6.2307,
                                   16.622, -5.6788, -0.0027,
                                   8.5869])
            q_simSlaveNormalized = np.array([0.5, 0.5, 0.64, 0.0,
                                             0.89, 0.43, 0.98,
                                             0.78, 0.42, 0.43,
                                             0.91])
        # elif q_in_ != 0 and q_in_norm == 0:
        elif (type(q_in_) != int) and (type(q_in_norm) == int):
            q_simSlave = q_in_.copy()
            q_simSlaveNormalized = []
            for i_ in range(len(q_simSlave)):
                if type(q_in_) is list:
                    q_in_ = np.array(q_in_)
                if (i_ == 4) or (i_ == 7):  # Invert: TR, TL
                    q_simSlaveNormalized = np.append(
                        q_simSlaveNormalized,
                        np.interp(
                            q_simSlave[i_],
                            (self.axisMins[i_],
                             self.axisMaxs[i_]), (1, 0)))
                elif (i_ == 6):  # Invert: BR
                    q_simSlaveNormalized = np.append(
                        q_simSlaveNormalized,
                        np.interp(
                            q_simSlave[i_],
                            (self.axisMins[i_],
                             self.axisMaxs[i_]), (1, 0)))
                else:
                    q_simSlaveNormalized = np.append(
                        q_simSlaveNormalized,
                        np.interp(
                            q_simSlave[i_],
                            (self.axisMins[i_],
                             self.axisMaxs[i_]), (0, 1)))
        elif (type(q_in_) == int) and (type(q_in_norm) != int):
            if type(q_in_) is list:
                q_in_ = np.array(q_in_)
            q_simSlaveNormalized = q_in_norm.copy()
            for i_ in range(len(q_simSlaveNormalized)):
                q_simSlave = np.interp(
                    q_simSlaveNormalized[i_],
                    (0, 1),
                    (self.axisMins[i_],
                     self.axisMaxs[i_]))
        else:
            if type(q_in_) is list:
                q_in_ = np.array(q_in_)
            if type(q_in_norm) is list:
                q_in_norm = np.array(q_in_norm)
            q_simSlave = q_in_.copy()
            q_simSlaveNormalized = q_in_norm.copy()
        self.axisCurrVal = q_simSlave.copy()
        self.axisCurrValNormalized = q_simSlaveNormalized.copy()
        if printOpt_:
            print('Axis    \t AXmea  \tNorm')
            for i_ in range(0, self.NumberAxis):
                print(slaveIndexLabels[i_], '\t',
                      slaveIndexSocket[i_], '   \t',
                      np.round(self.axisCurrVal[i_], 4), '   \t',
                      np.round(self.axisCurrValNormalized[i_], 2))

    # --------------------------------------------------------------------------
    # ------------------------ SENDING messages
    # --------------------------------------------------------------------------
    def sendCMDmsg2STRAS(self, msgtype2send, replyMode, xtraV, printOpt=False):
        """Low level send message to STRAS socket. Message locked to 33 bytes.

        Args:
            msgtype2send (int): Check the dictionary of message type.
            replyMode (int): If a answer is requested, 0 or 1.
            xtraV (str): If extended state or axis infos is requested.
            printOpt (bool, optional): Verbose. Defaults to False.

        Returns:
            bytes: If no error has been found, else returns 0
        """
        # ------------------------ Send a CMD message to the STRAS platform
        # ------------------------ see commandByCMD dictionary
        mesType = 0
        if printOpt:
            print('Sending message of type:\t', messageTypeDic[mesType])
        # ------------------------ Build message
        m2stras_syn = bytearray([0x21, 0x2a, 0x2a, 0x21])  # Synchro 0:4
        m2stras_1 = bytearray([0x21, 0x0, 0x0, 0x0])  # MessageLength 4:8
        m2stras_2 = int(0).to_bytes(4, "little", signed=False)  # MessType 8:12
        m2stras_3 = bytearray([0x1, 0x0, 0x0, 0x0])  # MessageVersion 12:16
        m2stras_4 = int(0).to_bytes(4, "little",
                                    signed=False)  # MsgRfrnce16:20
        m2stras_5 = int(replyMode).to_bytes(4, "little",
                                            signed=False)  # MesReplyMode 20:24
        m2stras_6 = int(msgtype2send).to_bytes(8, "little",
                                               signed=False)  # Message 24:32
        if xtraV != '':
            m2stras_7 = int(xtraV).to_bytes(4, "little", signed=False)
            pre_msg2stras = m2stras_syn + m2stras_1 + m2stras_2 + m2stras_3 \
                + m2stras_4 + m2stras_5 + m2stras_6 + m2stras_7
        else:
            pre_msg2stras = m2stras_syn + m2stras_1 + m2stras_2 + m2stras_3 \
                + m2stras_4 + m2stras_5 + m2stras_6

        # ------------------------ Do checksum
        if self.cycle < 0:
            m2stras_checksum = bytearray([0x1])
        else:
            # print('Chcksm:\t', sum(pre_msg2stras), sum(pre_msg2stras) % 256)
            m2stras_checksum = int(sum(pre_msg2stras) % 256
                                   ).to_bytes(1, "big", signed=False)
        msg2stras = pre_msg2stras + m2stras_checksum  # Message + checksum
        try:
            self.STRASslave.sendall(msg2stras)  # Send message
        except Exception:
            self.errorSendMsg = True
        try:
            data = self.STRASslave.recvmsg(4096)
            return data
        except socket.timeout:
            return 'err:timeout'
        except Exception:
            self.errorRecvMsg = True
            print('Couldnt receive data! - INCOMPATIBLE MESSAGE')
            return 0

    def send_VELIDX(self, axes_vals, printOpt_):
        """Adds constant velocitity to the STRAS slave DOF until set to 0 or
        reach limits.

        Args:
            axes_vals (list): List of list of the velocities to be sent to the
            STRAS slave socket [[axis_number0, velocity to send0],
            [axis_number1, velocity to send1], ...].
            printOpt_ (bool): Verbose on terminal
        """
        if self.port == 6666 or self.port == '6666':
            # ------------------------ Send a CMD message to the STRAS platform
            # ------------------------ see commandByCMD dictionary
            print('***SENDING VELOCITY***')
            num_axis = len(axes_vals)
            mesType = 0
            replyMode = 0
            mssLen = 28 + 4 + 1 + num_axis*8 + num_axis*2
            # 24 for the first 6 bytes (Synchro, Length, Type, Version,
            # Reference and Reply mode) + 1 Byte Checksum
            # + 4 (command) + 8*number of axis (double value)
            # + 2*number of axis(short) + 4 CMD_MESSAGE
            if printOpt_:
                print('Sending message of type:\t', messageTypeDic[mesType],
                      'with len', mssLen)
            # ------------------------ Build message
            m2stras_syn = bytearray([0x21, 0x2a, 0x2a, 0x21])  # Synchro 0:4
            m2stras_1 = int(mssLen).to_bytes(4, "little",
                                             signed=False)  # MessageLength 4:8
            m2stras_2 = int(0).to_bytes(
                4, "little", signed=False)  # MessType 8:12
            m2stras_3 = bytearray([0x1, 0x0, 0x0, 0x0])  # MessageVersion 12:16
            m2stras_4 = int(0).to_bytes(4, "little",
                                        signed=False)  # MsgRfrnce 16:20
            m2stras_5 = int(replyMode).to_bytes(
                4, "little", signed=False)  # MesReplyMode 20:24
            m2stras_6 = int(52).to_bytes(4, "little",
                                         signed=False)  # MessageType 24:28
            m2stras_7 = int(num_axis+num_axis*8+1).to_bytes(
                4, "little", signed=False)  # Lenght of bytes sent 28:32
            num_axis_byte = int(num_axis).to_bytes(1, "little",
                                                   signed=False)  # Idx 32
            pre_msg2stras = m2stras_syn + m2stras_1 + m2stras_2 + m2stras_3 \
                + m2stras_4 + m2stras_5 + m2stras_6 + m2stras_7 + num_axis_byte
            for i in range(num_axis):
                ax_ = int(axes_vals[i][0]).to_bytes(1,
                                                    "little",
                                                    signed=False)  # Index 33
                vel_ = pack('<d', axes_vals[i][1])  # Value 34:42
                pre_msg2stras = pre_msg2stras + ax_ + vel_
                if printOpt_:
                    print('Axes ', axes_vals[i][0],
                          ' increasing by: ', axes_vals[i][1])
            # ------------------------ Do checksum
            if self.cycle < 0:
                m2stras_checksum = bytearray([0x1])
            else:
                m2stras_checksum = int(sum(pre_msg2stras) % 256
                                       ).to_bytes(1, "big", signed=False)
            msg2stras = pre_msg2stras + m2stras_checksum  # Message + checksum
            if printOpt_:
                print('Len of MSG:\t'+str(int(len(msg2stras)))+'\n',
                      msg2stras)

            try:
                self.STRASslave.sendall(msg2stras)  # Send message
                print('***\tVELOCITY SENT\t***')
            except Exception:
                self.errorSendMsg = True
                self.errors = 'ErrorVelIDX'
                print('!!!!!!\tCOULD NOT SENT\t!!!!!!')
            self.cycle = self.cycle + 1
            # try:
            #     data = self.STRASslave.recvmsg(4096)
            #     return data
            # except socket.timeout:
            #     print('Timeout error!')
            # except Exception:
            #     self.errorRecvMsg = True
            #     print('Couldnt receive data! - UNKNOWN')
            #     return 0
        else:
            print('USING SLAVE, CHANGE PORT TO 6666 PLEASE!')

    def send_GoToPosPerAxis(self, listOfListByAxes_, printOpt_):
        """Send a comand to reach certain position. Low level controller handles
        separetly each degree of freedom, meaning only 1 DOF moves at a time.

        Args:
            listOfListByAxes_ (list): List of list of positions to be set by
            each DOF. [[axis_number0, axis_position0, axis_rel-speed0],
            [axis_number1, axis_position1, axis_rel-speed1]]
            printOpt_ (bool): Verbose.
        """
        if self.port == 6666 or self.port == '6666':
            # ------------------------ Send a CMD message to the STRAS platform
            # ------------------------ see commandByCMD dictionary
            # ------------------------ listOfListByAxes[axis, pos, rl_max_vel]
            if printOpt_:
                print('***GO TO POS***')
            num_axis = len(listOfListByAxes_)
            mesType = 0
            replyMode = 0
            mssLen = 28 + 4 + 1 + 1*num_axis + 8*num_axis + 8*num_axis + 1
            # 28 'syncr','len','type','vrsion','refnce','rplymode','cmndCOMAND'
            # 4  'lenOfInfo'
            # 1  'numIndex'
            # 1  'index'
            # 8  'maxVel'
            # 8  'position'

            if printOpt_:
                print('Sending message of type:\t', messageTypeDic[mesType],
                      'with len', mssLen)
            # ------------------------ Build message
            m2stras_syn = bytearray([0x21, 0x2a, 0x2a, 0x21])  # Synchro 0:4
            m2stras_1 = int(mssLen).to_bytes(
                4, "little", signed=False)  # MessageLength 4:8
            m2stras_2 = int(0).to_bytes(
                4, "little", signed=False)  # MessType 8:12
            m2stras_3 = bytearray([0x1, 0x0, 0x0, 0x0])  # MessageVersion 12:16
            m2stras_4 = int(0).to_bytes(
                4, "little", signed=False)  # MsgRfrc 16:20
            m2stras_5 = int(replyMode).to_bytes(
                4, "little", signed=False)  # MesReplyMode 20:24
            m2stras_6 = int(42).to_bytes(
                4, "little", signed=False)  # MessageType 24:28
            m2stras_7 = int(num_axis+num_axis*8*2+1).to_bytes(
                4, "little", signed=False)  # Len of CMD commands 28:32
            num_axis_byte = int(num_axis).to_bytes(1, "little",
                                                   signed=False)  # Idx 33
            pre_msg2stras = m2stras_syn + m2stras_1 + m2stras_2 + m2stras_3 \
                + m2stras_4 + m2stras_5 + m2stras_6 + m2stras_7 + num_axis_byte
            for i in range(num_axis):
                ax_ = int(listOfListByAxes_[i][0]).to_bytes(
                    1, "little", signed=False)  # Axs 1Byte
                pos_ = pack('<d', listOfListByAxes_[i][1])  # Value 8Bytes
                max_vel_ = pack('<d', listOfListByAxes_[i][2])  # Value 8Bytes
                pre_msg2stras = pre_msg2stras + ax_ + pos_ + max_vel_
                if printOpt_:
                    print('Axis ',
                          listOfListByAxes_[i][0],
                          ' moving to: ',
                          np.round(listOfListByAxes_[i][1], 2),
                          ' at a max relative velocity ',
                          np.round(listOfListByAxes_[i][2], 2))
            # ------------------------ Do checksum
            if self.cycle < 0:
                m2stras_checksum = bytearray([0x1])
            else:
                m2stras_checksum = int(sum(pre_msg2stras) % 256
                                       ).to_bytes(1, "big", signed=False)
            msg2stras = pre_msg2stras + m2stras_checksum  # Message + checksum
            if printOpt_:
                print('Len of MSG:\t'+str(int(len(msg2stras)))+'\n',
                      msg2stras)

            try:
                self.STRASslave.sendall(msg2stras)  # Send message
                if printOpt_:
                    print('***\tSENT GOTOIDX\t***')
            except Exception:
                self.errorSendMsg = True
                self.errors = 'ErrorVelIDX'
                print('!!!!!!\tCOULD NOT SENT\t!!!!!!')
            # try:
            #     data = self.STRASslave.recvmsg(4096)
            #     return data
            # except socket.timeout:
            #     print('Timeout error!')
            # except Exception:
            #     self.errorRecvMsg = True
            #     print('Couldnt receive data! - UNKNOWN')
            #     return 0
        else:
            print('USING SLAVE, CHANGE PORT TO 6666 PLEASE!')

    def send_GoToPos(self, listOfListByAxes_, printOpt_):
        """Send a comand to reach certain position. Low level controller handles
        separetly each degree of freedom, meaning only 1 DOF moves at a time

        Args:
            listOfListByAxes_ (list): List of list of positions to be set by
            each DOF. [[axis_number0, axis_position0, axis_rel-speed0],
            [axis_number1, axis_position1, axis_rel-speed1]]
            printOpt_ (bool): Verbose
        """
        if self.port == 6666 or self.port == '6666':
            # ------------------------ Send a CMD to the STRAS platform
            # ------------------------ see commandByCMD dictionary
            # ------------------------ listOfListByAxes[axs, pos, rel_maxvel]
            if printOpt_:
                print('***GO TO POS***')
            num_axis = len(listOfListByAxes_)
            mesType = 0
            replyMode = 0
            mssLen = 28 + 4 + 1 + 1*num_axis + 8*num_axis + 8*num_axis + 1
            # 28 'sync','len','type','vrsion','refnce','rplymode','cmdCOMAND'
            # 4  'lenOfInfo'
            # 1  'numIndex'
            # 1  'index'
            # 8  'maxVel'
            # 8  'position'

            if printOpt_:
                print('Sending message of type:\t', messageTypeDic[mesType],
                      'with len', mssLen)
            # ------------------------ Build message
            m2stras_syn = bytearray([0x21, 0x2a, 0x2a, 0x21])  # Synchro 0:4
            m2stras_1 = int(mssLen).to_bytes(4, "little",
                                             signed=False)  # MsgLength 4:8
            m2stras_2 = int(0).to_bytes(
                4, "little", signed=False)  # MessType 8:12
            m2stras_3 = bytearray([0x1, 0x0, 0x0, 0x0])  # MsgVersion 12:16
            m2stras_4 = int(0).to_bytes(
                4, "little", signed=False)  # MsgRfrc 16:20
            m2stras_5 = int(replyMode).to_bytes(
                4, "little", signed=False)  # MesReplyMode 20:24
            m2stras_6 = int(620).to_bytes(4, "little",
                                          signed=False)  # MessageType 24:28
            m2stras_7 = int(num_axis+num_axis*8*2+1).to_bytes(
                4, "little", signed=False)  # Len of CMD commands 28:32
            num_axis_bt = int(num_axis).to_bytes(1, "little",
                                                 signed=False)  # Idx 33
            pre_msg2stras = m2stras_syn + m2stras_1 + m2stras_2 + m2stras_3 \
                + m2stras_4 + m2stras_5 + m2stras_6 + m2stras_7 + num_axis_bt
            for i in range(num_axis):
                ax_ = int(listOfListByAxes_[i][0]).to_bytes(
                    1, "little", signed=False)  # Axs 1Byte
                pos_ = pack('<d', listOfListByAxes_[i][1])  # Value 8Bytes
                max_vel_ = pack('<d', listOfListByAxes_[i][2])  # Value 8Btes
                pre_msg2stras = pre_msg2stras + ax_ + pos_ + max_vel_
                if printOpt_:
                    print('Axis ',
                          listOfListByAxes_[i][0],
                          ' moving to: ',
                          np.round(listOfListByAxes_[i][1], 2),
                          ' at a max relative velocity ',
                          np.round(listOfListByAxes_[i][2], 2))
            # ------------------------ Do checksum
            if self.cycle < 0:
                m2stras_checksum = bytearray([0x1])
            else:
                # print(
                #     'Chcksm:',
                #     sum(pre_msg2stras), sum(pre_msg2stras) % 256)
                m2stras_checksum = int(sum(pre_msg2stras) % 256
                                       ).to_bytes(1, "big", signed=False)
            msg2stras = pre_msg2stras + m2stras_checksum  # Message + chcksum
            if printOpt_:
                print('Len of MSG:\t'+str(int(len(msg2stras)))+'\n',
                      msg2stras)

            try:
                self.STRASslave.sendall(msg2stras)  # Send message
                if printOpt_:
                    print('***\tSENT GOTOIDX\t***')
            except Exception:
                self.errorSendMsg = True
                self.errors = 'ErrorVelIDX'
                print('!!!!!!\tCOULD NOT SENT\t!!!!!!')
            # try:
            #     data = self.STRASslave.recvmsg(4096)
            #     return data
            # except socket.timeout:
            #     print('Timeout error!')
            # except Exception:
            #     self.errorRecvMsg = True
            #     print('Couldnt receive data! - UNKNOWN')
            #     return 0
        else:
            print('USING SLAVE, CHANGE PORT TO 6666 PLEASE!')

    def send_GoToPos_byLabel(self, listOfListByAxes_lbl, printOpt_):
        """The same command as send_GoToPos but uses labels instead of axis
        index number.

        Args:
            listOfListByAxes_lbl (list): List of list containing the label of
            the DOF to control, position and maximum relative velocity.
            [[axis_label0, axis_position0, axis_rel-speed0],
            [axis_label1, axis_position1, axis_rel-speed1]]
            printOpt_ (bool): Verbose.
        """
        output_ = []
        for k_ in (listOfListByAxes_lbl):
            idx_ = dicSlaveIndex[k_[0]]
            pos_ = k_[1]
            max_rel_vel_ = k_[2]
            # For boundaries on the bending then inside the channel
            if idx_ == 'BR':
                if self.axisCurrValNormalized[dicSlaveIndex['TR'] > 0.45]:
                    output_.append([idx_, pos_, max_rel_vel_])
                else:
                    print('Right tool inside the channel....')
                    pass
            elif idx_ == 'BL':
                if self.axisCurrValNormalized[dicSlaveIndex['TL'] > 0.45]:
                    output_.append([idx_, pos_, max_rel_vel_])
                else:
                    print('Left tool inside the channel....')
                    pass
            elif idx_ != 'BR' and idx_ != 'BL':
                # idxInSlave_ = [kk_ for kk_ in
                #                range(len(slaveOrder))
                #                if slaveOrder[kk_] == k_[0]][0]
                # if ((self.axisCurrVal[idxInSlave_] <
                #     self.axisMaxs[idxInSlave_]) and
                #     (self.axisCurrVal[idxInSlave_] >
                #      self.axisMins[idxInSlave_])):
                output_.append([idx_, pos_, max_rel_vel_])
                # else:
                #     print('Requested position outside of boundaries:',
                #           idx_, 'to', pos_)
            # if printOpt_:
            # print(
            #     'CMD2STRAS',
            #     k_[0],
            #     np.round(k_[1], 2),
            #     k_[2])
        self.send_GoToPos(output_, printOpt_)

    # --------------------------------------------------------------------------
    # ------------------------ RECEIVING/DECODING messages
    # --------------------------------------------------------------------------
    def decodeMsg(self, data, expectedReply, printOpt):
        """Decodes a byte message from an reply from the socket.

        Args:
            data (bytes): bytes received from the socket.
            expectedReply (int): If a reply is expected to decode a longer
            message.
            printOpt (bool): Verbose.
        """
        # ------------------------ Decode a message dependant of the expected
        # ------------------------ reply from previos message sent
        # ------------------------ see commandByCMD dictionary
        if type(data) == 'int':
            print(data)
            print('^^ Check message received! ^^')
        else:
            r_STRAS = data[0]
            r_STRAS_syn = r_STRAS[:][:4].decode()  # Synchro
            r_STRAS_len = int.from_bytes(r_STRAS[4:8],
                                         "little", signed=False)  # Length
            r_STRAS_type = int.from_bytes(r_STRAS[8:12],
                                          "little", signed=False)  # Type
            # r_STRAS_mes_ver = int.from_bytes(r_STRAS[12:16],
            #                                  "little",
            #                                  signed=False)  # Version
            # r_STRAS_mes_ref = int.from_bytes(r_STRAS[16:20],
            #                                  "little",
            #                                  signed=False)  # Reference
            # r_STRAS_rep_mode = int.from_bytes(r_STRAS[20:24],
            #                                   "little",
            #                                   signed=False)  # RplyMode
            # ------------------------ If synchro is OFF breaks the loop
            if r_STRAS_syn == '!**!' and self.cycle == 0:
                if printOpt:
                    print('Synchro OK ...')
                    print('INCOMING Message type',
                          messageTypeDic[r_STRAS_type],
                          'with len\t', r_STRAS_len-24-1, 'bytes')
            elif r_STRAS_syn != '!**!' and self.cycle == 0:
                print('Check Synchro!')
            if expectedReply == 71:
                STRASslave.msgCMD_AXIS_infos(self, r_STRAS, printOpt)
            if expectedReply == 7:
                STRASslave.msgCMD_RECV_STATE(self, r_STRAS, printOpt)
            if expectedReply == 410:
                STRASslave.msgCMD_RECV_STATE(self, r_STRAS, printOpt)
            # if expectedReply == 3:
            #     STRASslave.msgCMD_RECV_STATE(self, r_STRAS, printOpt)

    def msgCMD_AXIS_infos(self, r_STRAS, printOpt):
        """Decodes axis information message from the STRAS slave socket.

        Args:
            r_STRAS (bytes): A byte response from the socket, previously asked
            for axis infos.
            printOpt (bool): Verbose.
        """
        # ------------------------ Decode AXIS infos message
        # ------------------------ Get axis infos
        if printOpt:
            print(commandByCMD[71], len(r_STRAS)-24-1)
            print('RAW LEN:\t', len(r_STRAS), 'bytes')
        # New main message unpack
        self.NumberAxis = int.from_bytes(
            r_STRAS[24:25], "little", signed=False)
        self.ActiveAxis = int.from_bytes(
            r_STRAS[25:26], "little", signed=False)
        if printOpt:
            print('NumberOfAxisRobot:\t', self.NumberAxis)
            print('NumberOfActiveAxisRobot:\t', self.ActiveAxis)
        init_axis_info = 26
        buffer_size = 17
        for i in range(0, self.NumberAxis):
            ix_1 = init_axis_info+(i*buffer_size)
            active_axis = int.from_bytes(r_STRAS[ix_1:ix_1+1],
                                         "little", signed=False)
            min_per_axis = unpack('<d', r_STRAS[ix_1+1:ix_1+9])
            max_per_axis = unpack('<d', r_STRAS[ix_1+9:ix_1+17])
            self.axisMins.append(min_per_axis[0])
            self.axisMaxs.append(max_per_axis[0])
            if printOpt:
                print(slaveIndexLabels[i], '', slaveIndexSocket[i], '\t',
                      ('ACTIVE' if active_axis > 0 else 'INACTIVE'))
                print('\tmin: ', min_per_axis)
                print('\tmax: ', max_per_axis)

    def msgCMD_RECV_STATE(self, r_STRAS, printOpt, extendedState=False):
        """Receive a message from the STRAS robot, expected a state

        Args:
            r_STRAS (bytes): Byte response from requested a state.
            printOpt (bool): Verbose.
            extendedState (bool, optional): If an extended state is expected.
            Defaults to False.
        """
        # ------------------------ Receive message depending on the expected
        # ------------------------ reply from previos message sent
        if printOpt:
            print(commandByCMD[7])
            # print('RAW LEN:\t', len(r_STRAS), 'bytes')
            print('MSG LEN:\t', len(r_STRAS)-24-1)
        self.tick = int.from_bytes(r_STRAS[24:32],
                                   "little",
                                   signed=False)  # Int64-8Bits
        self.controllerMode = int.from_bytes(r_STRAS[32:33], "little",
                                             signed=False)
        self.ContErr = int.from_bytes(r_STRAS[33:34], "little", signed=False)
        self.TraceOn = int.from_bytes(r_STRAS[34:35], "little", signed=False)
        self.CalibratedRobot = int.from_bytes(r_STRAS[35:36],
                                              "little", signed=False)
        self.SecurityOnRobot = int.from_bytes(r_STRAS[36:37], "little",
                                              signed=False)
        self.PowerOnRobot = int.from_bytes(r_STRAS[37:38], "little",
                                           signed=False)
        self.DcOnRobot = int.from_bytes(r_STRAS[38:39], "little",
                                        signed=False)
        self.NumberAxis = int.from_bytes(
            r_STRAS[39:40], "little", signed=False)
        self.ActiveAxis = int.from_bytes(
            r_STRAS[40:41], "little", signed=False)
        if printOpt:
            print('Controller Mode:\t', self.controllerMode)
            # print('Controller Mode:\t', r_STRAS_ContMode, '\t',
            #       controllerModeOps[r_STRAS_ContMode])
            print('NumberOfAxisRobot:\t', self.NumberAxis)
            print('NumberOfActiveAxisRobot:\t',
                  self.ActiveAxis)

        init_axis_info = 41
        buffer_size_axis = 27
        last_ix = init_axis_info
        if printOpt:
            print('Axis    \t AXmea  \t AXref  \t ME-REF  \tNorm')
        if self.cycle > 0:
            # self.axisPastVal = self.axisCurrVal.copy()
            # self.axisPastValNorm = self.axisCurrValNormalized.copy()
            # self.axisEnable = []
            # self.axisCalibrated = []
            # self.axisUsed = []
            self.axisCurrVal = np.array([], dtype=np.float32)
            # self.axisCurrValVel = np.array([], dtype=np.float32)
            self.axisCurrValNormalized = np.array([], dtype=np.float32)
            # self.axisReference = []
            # self.axisTorque = []
            # self.axisButtons = []
        for i in range(0, self.NumberAxis):
            ix_1 = last_ix+(i*buffer_size_axis)
            ax_enab = unpack('<?', r_STRAS[ix_1:ix_1+1])
            ax_cal = unpack('<?', r_STRAS[ix_1+1:ix_1+2])
            ax_used = unpack('<?', r_STRAS[ix_1+2:ix_1+3])
            ax_meas = unpack('<d', r_STRAS[ix_1+3:ix_1+3+8])
            ax_ref = unpack('<d', r_STRAS[ix_1+3+8:ix_1+3+8+8])
            # ax_torq = unpack('<d', r_STRAS[ix_1+3+8+8:ix_1+3+8+8+8])

            self.axisEnable.append(ax_enab[0])
            self.axisCalibrated.append(ax_cal[0])
            self.axisUsed.append(ax_used[0])
            # self.axisCurrVal.append(ax_meas[0])
            self.axisCurrVal = np.append(self.axisCurrVal, ax_meas[0])
            self.axisReference.append(ax_ref[0])
            # self.axisTorque.append(ax_torq[0])
            if (i == 4) or (i == 7):  # Invert: TR, TL
                self.axisCurrValNormalized = np.append(
                    self.axisCurrValNormalized,
                    np.interp(ax_meas[0],
                              (self.axisMins[i],
                              self.axisMaxs[i]), (1, 0)))
            elif (i == 6):  # Invert: BR
                self.axisCurrValNormalized = np.append(
                    self.axisCurrValNormalized,
                    np.interp(ax_meas[0],
                              (self.axisMins[i],
                              self.axisMaxs[i]), (1, 0)))
            else:
                self.axisCurrValNormalized = np.append(
                    self.axisCurrValNormalized,
                    np.interp(ax_meas[0],
                              (self.axisMins[i],
                              self.axisMaxs[i]), (0, 1)))

            if i == self.NumberAxis-1:
                last_ix = ix_1+3+8+8+8
            if printOpt:
                print(slaveIndexLabels[i], '\t', slaveIndexSocket[i], '   \t',
                      np.round(ax_meas[0], 4),
                      '   \t', np.round(ax_ref[0], 2),
                      '   \t', np.round(ax_meas[0]-ax_ref[0], 2),
                      '   \t', np.round(self.axisCurrValNormalized[-1], 2))

        r_STRAS_NBut = int.from_bytes(r_STRAS[last_ix:last_ix+1],
                                      "little", signed=False)
        if printOpt:
            print('NumberOfButtons:\t', r_STRAS_NBut)
        last_ix = last_ix + 1
        for k in range(0, r_STRAS_NBut):
            ix_1 = last_ix+(k*1)
            buttonLvl = int(unpack('<?', r_STRAS[ix_1:ix_1+1])[0])
            self.axisButtons.append(buttonLvl)
            if k == r_STRAS_NBut-1:
                last_ix = ix_1 + 1
        if self.cycle > 1 and len(self.axisButtons) > r_STRAS_NBut:
            self.axisButtons = self.axisButtons[-r_STRAS_NBut:]
        if printOpt and r_STRAS_NBut > 0:
            print('Buttons', self.axisButtons)
        r_STRAS_NumDDiom = int.from_bytes(r_STRAS[last_ix:last_ix+1],
                                          "little", signed=False)
        last_ix = last_ix + 1
        r_STRAS_DDiom = []
        for i in range(0, r_STRAS_NumDDiom):
            ix_1 = init_axis_info+(i*r_STRAS_NumDDiom)
            lvlDDio = int(unpack('<?', r_STRAS[ix_1:ix_1+1])[0])
            r_STRAS_DDiom.append(lvlDDio)
            if i == r_STRAS_NumDDiom - 1:
                last_ix = ix_1+1
        # r_STRAS_Ngroups = int.from_bytes(r_STRAS[last_ix:last_ix+1],
        #                        git          "little", signed=False)  # Usable
        # print(self.axisCurrVal)
        last_ix = last_ix + 1
        # print('NumberOfGroups:\t', r_STRAS_Ngroups)

        # Keep only last records
        self.axisCurrVal = self.axisCurrVal[-12:]
        self.axisCurrValNormalized = self.axisCurrValNormalized[-12:]
        self.axisCurrValVel = self.axisCurrValVel[-12:]
        self.axisPastVal = self.axisPastVal[-12:]
        self.axisEnable = self.axisEnable[-12:]
        self.axisCalibrated = self.axisCalibrated[-12:]
        self.axisUsed = self.axisUsed[-12:]
        self.axisReference = self.axisReference[-12:]

        if extendedState:
            # r_STRAS_refTeleOp = unpack('<d', r_STRAS[last_ix:last_ix+8])
            last_ix = last_ix + 8
            # r_STRAS_ax_lck_dic = {0: 'UNLOCKED', 1: 'LOCKED', 2: 'ND'}
            # r_STRAS_ax_lck = int.from_bytes(r_STRAS[last_ix:last_ix+1],
            #                                 "little", signed=False)
        # print('Loop count:\t'+str(f+1))

    # --------------------------------------------------------------------------
    # ------------------------ EASY COMMANDS TO USE
    # --------------------------------------------------------------------------
    def getAxisInfo(self, printOpt_):
        """A second layer of the axis info sending/receiving of commands.

        Args:
            printOpt_ (bool): Verbose.
        """
        # ------------------------ To ease of use
        r_STRAS = self.sendCMDmsg2STRAS(msgtype2send=71,
                                        replyMode=1,
                                        xtraV='',
                                        printOpt=printOpt_)
        self.decodeMsg(r_STRAS, expectedReply=71, printOpt=printOpt_)
        self.cycle = self.cycle + 1

    def getState(self, printOpt_):
        """A second layer of the axis info sending/receiving of commands. Easier
        way to adquire a state.

        Args:
            printOpt_ (bool): Verbose.
        """
        # ------------------------ To ease of use
        r_STRAS = self.sendCMDmsg2STRAS(msgtype2send=7,
                                        replyMode=1,
                                        xtraV='',
                                        printOpt=printOpt_)
        if r_STRAS == 'err:timeout':
            print('Timeout Error in Slave...\n')
            pass
        elif type(r_STRAS) == tuple and len(r_STRAS) != 0:
            self.decodeMsg(r_STRAS, expectedReply=7, printOpt=printOpt_)
            if self.port == 6667:
                for i in range(len(self.axisCurrVal)):
                    if i in range(7):
                        # To update min and max JUST IN CASE - CALIBRATION
                        if self.axisCurrVal[i] > self.axisMaxs[i]:
                            self.axisMaxs[i] = self.axisCurrVal[i]
                        if self.axisCurrVal[i] < self.axisMins[i]:
                            self.axisMins[i] = self.axisCurrVal[i]
                self.cycle = self.cycle + 1
        else:
            print('DATA RECEIVED NOT USEFUL!\n', r_STRAS)

    def easyStart(self):
        """Easy command to start the socket.
        """
        self.setTimeOut(1.0)
        self.connect(True)
        self.getAxisInfo(False)

    # --------------------------------------------------------------------------
    # ------------------------ For visualization
    # --------------------------------------------------------------------------
    # def visualizeState(self, printOpt_=False):
    #     """Function to visualize the current state, it changes from the state
    #     on master range and order to the control space.

    #     Args:
    #         printOpt_ (bool, optional): Verbose. Defaults to False.

    #     Returns:
    #         np.array: a numpy array with the rendered robot in the camera image
    #         plane
    #     """
    #     img = self.visualizeFrame(
    #         map_norm2range(
    #             order_slave2qp(self.axisCurrValNormalized),
    #             orderRanges(qlims_down_joint_pos, qpOrder),
    #             orderRanges(qlims_up_joint_pos, qpOrder)))
    #     return img

    # def visualizeFrame(self, q2visualize, printOpt_=False):
    #     """Visualize the frame corresponding to q2visualize list-or-np.array in
    #     qp order and range.

    #     Args:
    #         q2visualize (list): List of DOF of the STRAS robot in qp form and
    #         in qp range.
    #         printOpt_ (bool, optional): Verbose. Defaults to False.

    #     Returns:
    #         np.array: a numpy array with the rendered robot in the camera image
    #         plane
    #     """
    #     if not(self.initializeRenderFlag):
    #         self.initializeVisualization(self, False)
    #     strasOutput_slave = self.robot_stras.computemodels(
    #         q2visualize.copy())
    #     Tr_e_slave = strasOutput_slave[0].copy()
    #     # Jr_e_slave = strasOutput_slave[1].copy()
    #     Tl_e_slave = strasOutput_slave[2].copy()
    #     # Jl_e_slave = strasOutput_slave[3].copy()
    #     Tr_0_slave = strasOutput_slave[4].copy()
    #     # Jr_0_slave = strasOutput_slave[5].copy()
    #     Tl_0_slave = strasOutput_slave[6].copy()
    #     # Jl_0_slave = strasOutput_slave[7].copy()
    #     Te_slave = strasOutput_slave[8].copy()
    #     # Je_slave = strasOutput_slave[9].copy()
    #     shpRslve, shapeLslave, tipend_slave = self.robot_stras.computeShape(
    #         q2visualize.copy())
    #     img_ = self.render.render_stras_arms(
    #         shpRslve, shapeLslave, tipend_slave, Tr_e_slave,
    #         Tl_e_slave, Tr_0_slave, Tl_0_slave, Te_slave)
    #     return img_

    # def initializeVisualization(self, printOpt_=False):
    #     """To initialize the visualization renders and else.

    #     Args:
    #         printOpt_ (bool, optional): Verbose. Defaults to False.
    #     """
    #     d = {}
    #     with open(configFile) as f:
    #         for line in f:
    #             (key, val) = line.split("=")
    #             d[key] = val

    #     # Create camera and robot model
    #     cam = camera_model(
    #         d['camera_calibration'][:-1])
    #     robot_right = robot_model(
    #         configfile=d['robot_config_right'][:-1])
    #     robot_left = robot_model(
    #         configfile=d['robot_config_left'][:-1])
    #     robot_end = robot_twobending(
    #         configfile=d['robot_config_endoscope'][:-1])

    #     # Create the overall stras robot model
    #     # (to compute pose and jacobian of each robot)
    #     self.robot_stras = StrasModel(robot_right, robot_left, robot_end)

    #     # Create rendering to render robot shape
    #     w, h = cam.getImSize()
    #     self.render = render_shape(w, h, self.robot_stras)
    #     self.render.setCamProperties(cam.getF(), cam.getC(), cam.K)

    #     # Update the initialization flag
    #     self.initializeRenderFlag = True

    # # --------------------------------------------------------------------------
    # # ------------------------ To control frame position
    # # --------------------------------------------------------------------------
    # # To clean the Jacobians and Transformation matrices
    # def clean_matrices(self):
    #     self.Tr_e = 0
    #     self.Jr_e = 0
    #     self.Tl_e = 0
    #     self.Jl_e = 0
    #     self.Tr_0 = 0
    #     self.Jr_0 = 0
    #     self.Tl_0 = 0
    #     self.Jl_0 = 0
    #     self.Te = 0
    #     self.Je = 0

    # # To control each frame to a target
    # def goToGeoPos(self, frames_vals, printOpt_, real_state=True):
    #     """To obtain the commands needed to achieve the goal stated on the
    #     inputs.

    #     Args:
    #         frames_vals (list): List of list containing the frame to control
    #         and the goal.
    #         printOpt_ (bool): Verbose
    #         real_state (bool, optional): If false it uses the last simulated
    #         stated declared. Defaults to True.

    #     Returns:
    #         list: New list of dof variables sent to the STRAS slave.
    #     """
    #     # ------------------------ Send a CMD message to the STRAS platform
    #     # ------------------------ see commandByCMD dictionary
    #     # ------------------------ frames_vals = [[frame, pos]]
    #     # Frames:   R_e, L_e: Right and Left tool tip in the camera frame
    #     #           R, L: Right and Left tool tip in the global frame
    #     #           E: Endoscope camera frame
    #     if self.modelsComputedFlag:
    #         if self.port == 6666 or self.port == '6666':
    #             if not(self.preliminariesForCompute):
    #                 self.initializeModels(False)
    #             if printOpt_:
    #                 print('***GO TO GEO POS***')
    #             if real_state:
    #                 self.getState(False)
    #             number_frames2control = len(frames_vals)
    #             frames = []
    #             goals = []
    #             for ix in range(number_frames2control):
    #                 frames.append(frames_vals[ix][0])
    #                 goals.append(frames_vals[ix][1])
    #             if printOpt_:
    #                 print('frames', frames)
    #                 print('goals', goals)
    #             q11_, q12_, q13_, q14_ = self.solveQP(frames, goals, printOpt_)
    #             # Outputing to self variables
    #             self.past_q = q11_.copy()
    #             self.current_q = q12_.copy()
    #             self.q_dot = q13_.copy()
    #             self.q_dot_labels = q14_.copy()
    #             # From qp normalized to slave ranges
    #             q_dot_pas_ = map_norm2slave_slaveOrder(
    #                 order_qp2slave(
    #                     normalize_q_qp(q11_.copy(),
    #                                    orderRanges(qlims_down_joint_pos,
    #                                                qpOrder),
    #                                    orderRanges(qlims_up_joint_pos,
    #                                                qpOrder))),
    #                 orderRanges(qlims_down_slave_pos, slaveOrder),
    #                 orderRanges(qlims_up_slave_pos, slaveOrder))
    #             q_dot_act_ = map_norm2slave_slaveOrder(
    #                 order_qp2slave(
    #                     normalize_q_qp(q12_.copy(),
    #                                    orderRanges(qlims_down_joint_pos,
    #                                                qpOrder),
    #                                    orderRanges(qlims_up_joint_pos,
    #                                                qpOrder))),
    #                 orderRanges(qlims_down_slave_pos, slaveOrder),
    #                 orderRanges(qlims_up_slave_pos, slaveOrder))
    #             res1_, flag1_ = checkInRange(
    #                 q_dot_act_,
    #                 orderRanges(qlims_down_slave_pos, slaveOrder),
    #                 orderRanges(qlims_up_slave_pos, slaveOrder))
    #             if flag1_:
    #                 print(res1_)
    #             cmd2Stras_qp = filter_q_dot(q_dot_pas_.copy(),
    #                                         q_dot_act_.copy())
    #             # print('cmd2Stras', cmd2Stras_qp)
    #             listOfListByAxcleares_ = []
    #             for u_ in range(len(cmd2Stras_qp)):
    #                 idxLbl_ = cmd2Stras_qp[u_][0]
    #                 idxVal_ = cmd2Stras_qp[u_][1]
    #                 idxInSlaveOrder = [
    #                     idx_ for idx_ in range(
    #                         len(slaveOrder)) if slaveOrder[idx_] == idxLbl_][0]
    #                 listOfListByAxcleares_.append(
    #                     [idxLbl_,
    #                         idxVal_,
    #                         0.05])
    #                 if printOpt_:
    #                     print(
    #                         'Step in', idxLbl_, ' joint values from',
    #                         np.round(self.axisCurrVal[idxInSlaveOrder], 3),
    #                         'to',
    #                         np.round(idxVal_, 3))
    #             if printOpt_:
    #                 print(listOfListByAxcleares_)
    #             return listOfListByAxcleares_
    #             # self.send_GoToPos_byLabel(
    #             #     listOfListByAxcleares_, printOpt_)
    #         else:
    #             print('USING SLAVE, CHANGE PORT TO 6666 PLEASE!')
    #     else:
    #         print('Initialize first to ensure correct location of target...')

    # # To initialize the robot models and activate the preliminaries flag
    # def initializeModels(self, printOpt_=False):
    #     """To initialize the models needed for the geometric control

    #     Args:
    #         printOpt_ (bool, optional): Verbose. Defaults to False.
    #     """
    #     self.dt_qp = 0.001
    #     rootGIT = 'stras_simulator/'
    #     configFile = 'datafolders.txt'
    #     path = rootPC_global+rootGITFolder_global+rootGIT
    #     d = {}
    #     with open(path+configFile) as f:
    #         for line in f:
    #             (key, val) = line.split("=")
    #             d[key] = val

    #     # Create camera and robot model
    #     robot_right = robot_model(
    #         configfile=path+d['robot_config_right'][:-1])
    #     robot_left = robot_model(
    #         configfile=path+d['robot_config_left'][:-1])
    #     robot_end = robot_twobending(
    #         configfile=path+d['robot_config_endoscope'][:-1])

    #     # Create the overall stras robot model
    #     # (to compute pose and jacobian of each robot)
    #     self.robot_stras = StrasModel(robot_right, robot_left, robot_end)
    #     self.preliminariesForCompute = True

    # # To compute and update the models (jacobians and transformation matrices)
    # def computeModels(self, slave_q_normalized_):
    #     """From a given state of the robot compute the models (transformation
    #     matrices and jacobians).

    #     Args:
    #         slave_q_normalized_ (list): A list of values corresponding to the
    #         DOF, needs to be normalized increasing [0, 1] with 0 being the min
    #         value.

    #     Returns:
    #         list: List of values of the DOF in qp range and order
    #     """
    #     # To check if slave_q_normalized_ is outside [0, 1]
    #     if True:
    #         res1, f1_ = checkInRange(slave_q_normalized_,
    #                                  [0]*len(slave_q_normalized_),
    #                                  [1]*len(slave_q_normalized_))
    #         if f1_:
    #             print('Normalized Slave Q outside [0, 1]', res1)
    #             return 0
    #     if not(self.preliminariesForCompute):
    #         self.initializeModels(False)

    #     q_in_qp_form_ = map_norm2range(
    #         order_slave2qp(slave_q_normalized_.copy()),
    #         orderRanges(qlims_down_joint_pos, qpOrder),
    #         orderRanges(qlims_up_joint_pos, qpOrder))

    #     # To check if q_in_qp_form_ is outside [lims_QP_down, lims_QP_up]
    #     if True:
    #         res2, f2_ = checkInRange(q_in_qp_form_,
    #                                  orderRanges(qlims_down_joint_pos,
    #                                              qpOrder),
    #                                  orderRanges(qlims_up_joint_pos,
    #                                              qpOrder))
    #         if f2_:
    #             print('q_in_qp_form_ outside [qpMin, qpMax]', res2)
    #             return 0

    #     strasOutput = self.robot_stras.computemodels(
    #         q_in_qp_form_)
    #     shape_R_, shape_L_, tipend_ = self.robot_stras.computeShape(
    #         q_in_qp_form_)
    #     self.Tr_e = strasOutput[0].copy()
    #     self.Jr_e = strasOutput[1].copy()
    #     self.Tl_e = strasOutput[2].copy()
    #     self.Jl_e = strasOutput[3].copy()
    #     self.Tr_0 = strasOutput[4].copy()
    #     self.Jr_0 = strasOutput[5].copy()
    #     self.Tl_0 = strasOutput[6].copy()
    #     self.Jl_0 = strasOutput[7].copy()
    #     self.Te = strasOutput[8].copy()
    #     self.Je = strasOutput[9].copy()
    #     self.modelsComputedFlag = True
    #     return q_in_qp_form_

    # # To solve the QP problem w.r.t. to the desired tarjet on each frame
    # def solveQP(self, frames2control_lbl, goals, printOpt_):
    #     """Solve the QP problem based on the expected goal position of the
    #     target

    #     Args:
    #         frames2control_lbl (list): List of frames to control.
    #         goals (list): List of goals to be achieved by each frame.
    #         printOpt_ (bool): Verbose.

    #     Returns:
    #         [type]: [description]
    #     """
    #     if type(goals) != 'numpy.ndarray':
    #         goals = np.array(goals)
    #     # ------------------------ Initialize variables for QP problem
    #     gammas = np.array([1, 1, 1, 1, 1])
    #     k = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    #     instant_q_inQPorderRange = self.computeModels(
    #         self.axisCurrValNormalized.copy())
    #     # print('instantQinqp', instant_q_inQPorderRange)
    #     # frames_available = ['R_e', 'R_0', 'L_e', 'L_0', 'E']
    #     # Complete Hessian Matrix
    #     H_total = np.zeros((10, 10))
    #     # Building the C
    #     C_total = np.zeros((10, 1))
    #     toPrint = []

    #     # ------------------------ Update the H (J.T*J) matriz and C vector
    #     # Activate the frame intented to control with these steps each time:
    #     #   Adquire frame position to control
    #     #   Adquire desired position location
    #     #   Compute the Hessian matrix from the Jacobian
    #     #       Add result to the Hessian_total of the tasks
    #     #   Update the pre_c vector
    #     #       Add result to the C_total of the tasks

    #     # The right tool
    #     if 'R_e' in frames2control_lbl or 'R_E' in frames2control_lbl \
    #        or 'r_e' in frames2control_lbl:
    #         frame2control = self.Tr_e[:3, -1]
    #         desiredLocation = goals[frames2control_lbl.index('R_e')]
    #         J_1 = np.zeros((9, 10))
    #         J_1[:3, :3] = self.Jr_e
    #         H_total += gammas[0]*np.dot(J_1.T, J_1)
    #         pre_c_1 = np.zeros((9, 1))
    #         pre_c_1[:3] = np.reshape(((
    #                 frame2control - desiredLocation)) / self.dt_qp, (3, 1))
    #         c_1 = k[0]*np.dot(J_1.T, pre_c_1)
    #         # c_1 = k[0]*(-pre_c_1.T@J_1).T  # WRONG
    #         C_total += gammas[0]*c_1
    #         toPrint = [0, 1, 2]
    #     if 'R_0' in frames2control_lbl or 'r_0' in frames2control_lbl:
    #         frame2control = self.Tr_0[:3, -1].copy()
    #         desiredLocation = goals[frames2control_lbl.index('R_0')]
    #         J_2 = np.zeros((9, 10))
    #         J_2[:3, :3] = self.Jr_0
    #         J_2[:3, 6:] = self.Je
    #         H_total += gammas[1]*np.dot(J_2.T, J_2)
    #         pre_c_2 = np.zeros((9, 1))
    #         pre_c_2[:3] = np.reshape(((
    #                 frame2control - desiredLocation)) / self.dt_qp, (3, 1))
    #         c_2 = k[1]*np.dot(J_2.T, pre_c_2)
    #         # c_2 = k[1]*(-pre_c_2.T@J_2).T  # WRONG
    #         C_total += gammas[1]*c_2
    #         toPrint = [0, 1, 2, 6, 7, 8, 9]
    #     # The left tool
    #     if 'L_e' in frames2control_lbl or 'L_E' in frames2control_lbl \
    #        or 'l_e' in frames2control_lbl:
    #         frame2control = self.Tl_e[:3, -1].copy()
    #         desiredLocation = goals[frames2control_lbl.index('L_e')]
    #         J_3 = np.zeros((9, 10))
    #         J_3[3:6, 3:6] = self.Jl_e.copy()
    #         H_total += gammas[2]*np.dot(J_3.T, J_3)
    #         pre_c_3 = np.zeros((9, 1))
    #         pre_c_3[3:6] = np.reshape(((
    #                 frame2control - desiredLocation)) / self.dt_qp, (3, 1))
    #         c_3 = k[2]*np.dot(J_3.T, pre_c_3)
    #         # c_3 = k[2]*(-pre_c_3.T@J_3).T  # WRONG
    #         C_total += gammas[2]*c_3
    #         toPrint = [3, 4, 5]
    #     if 'L_0' in frames2control_lbl or 'l_0' in frames2control_lbl:
    #         frame2control = self.Tl_0[:3, -1].copy()
    #         desiredLocation = goals[frames2control_lbl.index('L_0')]
    #         J_4 = np.zeros((9, 10))
    #         J_4[3:6, 3:6] = self.Jl_0.copy()
    #         J_4[3:6, 6:] = self.Je.copy()
    #         H_total += gammas[3]*np.dot(J_4.T, J_4)
    #         pre_c_4 = np.zeros((9, 1))
    #         pre_c_4[3:6] = np.reshape(((
    #                 frame2control - desiredLocation)) / self.dt_qp, (3, 1))
    #         c_4 = k[3]*np.dot(J_4.T, pre_c_4)
    #         # c_4 = k[3]*(-pre_c_4.T@J_4).T  # WRONG
    #         C_total += gammas[3]*c_4
    #         toPrint = [3, 4, 5, 6, 7, 8, 9]
    #     # The endoscope
    #     if 'e' in frames2control_lbl or 'E' in frames2control_lbl:
    #         frame2control = self.Te[:3, -1].copy()
    #         desiredLocation = goals[frames2control_lbl.index('E')]
    #         J_5 = np.zeros((9, 10))
    #         J_5[6:, 6:] = self.Je.copy()
    #         H_total += gammas[4]*np.dot(J_5.T, J_5)
    #         pre_c_5 = np.zeros((9, 1))
    #         pre_c_5[6:] = np.reshape(((
    #                 frame2control - desiredLocation)) / self.dt_qp, (3, 1))
    #         c_5 = k[4]*np.dot(J_5.T, pre_c_5)
    #         # c_5 = k[4]*(-pre_c_5.T@J_5).T  # WRONG
    #         C_total += gammas[4]*c_5
    #         toPrint = [6, 7, 8, 9]

    #     gamma_reg = 0.000000001
    #     H_total += np.eye(10)*gamma_reg  # Adding the regularization term

    #     # ------------------------ QP solver - For motion
    #     n = 10
    #     x = cp.Variable(n)
    #     x0 = cp.Variable(n)
    #     max_per_rang = 10
    #     h_ = np.vstack((lims_up_joint, lims_down_joint)).flatten()

    #     # boxLimsDown = (
    #     #     orderRanges(qlims_down_joint_pos, qpOrder) -
    #     #     instant_q_inQPorderRange)
    #     # boxLimsUp = (
    #     #     orderRanges(qlims_up_joint_pos, qpOrder) -
    #     #     instant_q_inQPorderRange)
    #     # cons_qp2 = [-x2 <= (boxLimsDown/self.dt_qp),
    #     #             x2 <= (boxLimsUp/self.dt_qp)]

    #     # Statement of the QP problem as a CVXPY problem and handling
    #     # unfeasible solutions
    #     prob0 = cp.Problem(
    #         cp.Minimize((1.0/2)*cp.quad_form(x0, H_total) + C_total.T @ x0))
    #     prob = cp.Problem(
    #         cp.Minimize((1.0/2)*cp.quad_form(x, H_total) + C_total.T @ x),
    #         constraints=[np.row_stack((np.eye(10), - np.eye(10)))
    #                      @ x <= (h_/100)*max_per_rang])
    #     try:
    #         prob0.solve(warm_start=True)
    #         prob.solve(warm_start=True)
    #         q_dot0 = x.value*self.dt_qp
    #         # q_dot = x.value*self.dt_qp
    #     except Exception:
    #         print('\nNo feasible solution found!\n')
    #         q_dot0 = np.zeros(10)

    #     # if x.value is None:
    #     #     print('No feasible solution found!')
    #     #     q_dot0 = np.zeros(10)
    #         # q_dot = np.zeros(10)

    #     # ------------------------ Adding the q_dot to q
    #     # print('q_dot:', q_dot)
    #     q_out_0 = (
    #         instant_q_inQPorderRange +
    #         q_dot0.flatten())

    #     if printOpt_:
    #         # print('qComplete', instant_q_inQPorderRange)
    #         print('qIN', np.round(instant_q_inQPorderRange[toPrint], 3))
    #         print('qOUT', np.round(q_dot0[toPrint], 3))
    #     #       (q_dot0.flatten()*self.dt_qp).tolist())
    #     # q_out_ = (
    #     #     instant_q_inQPorderRange + q_dot.flatten()*self.dt_qp).tolist()

    #     # Saving the value to further debugging
    #     self.qp_value = prob.value

    #     return instant_q_inQPorderRange,\
    #         q_out_0, q_dot0, np.array(qpOrder)[q_dot0 != 0]
