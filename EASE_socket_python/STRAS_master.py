import socket
import numpy as np
from struct import unpack
from math import radians
from useful import map_norm2range, orderRanges, order_slave2qp
from model_utilities.vtk_render import render_shape
from model_utilities.camera_model import camera_model
# from model_utilities.robot_piecewise import Robot_piecewise_curvature
from model_utilities.robot_model import robot_model, robot_twobending
from model_utilities.stras_model import StrasModel
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

qlims_down_joint_pos = {'BR': -np.pi/(2*18.5),  # deg/s
                        'RR': radians(2),  # Rotation in degrees/s
                        'TR': 2,  # Translation in mm/s
                        'BL': -np.pi/(2*18.5),  # deg/s
                        'RL': radians(2),  # Rotation in degrees/s
                        'TL': 2,  # Translation in mm/s
                        'EBH': -np.pi/(2*185),  # HorB-E
                        'EBV': -np.pi/(2*185),  # VertB-E
                        'ER': radians(2),  # Rotation ENDO in degrees/s
                        'ET': 0}  # Translation ENDO in mm/s
qlims_up_joint_pos = {'BR': np.pi/(2*18.5),  # deg/s
                      'RR': radians(248),  # Rotation in degrees/s
                      'TR': 72,  # Translation in mm/s
                      'BL': np.pi/(2*18.5),  # deg/s
                      'RL': radians(248),  # Rotation in degrees/s
                      'TL': 72,  # Translation in mm/s
                      'EBH': np.pi/(2*185),  # HorB-E
                      'EBV': np.pi/(2*185),  # VertB-E
                      'ER': radians(248),  # Rotation ENDO in degrees/s
                      'ET': 95}  # Translation ENDO in mm/s
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
               'EBV', 'EBH', 'ER', 'ET']
qpOrder = ['BR', 'RR', 'TR',
           'BL', 'RL', 'TL',
           'EBV', 'EBH', 'ER', 'ET']

masterIndexLabels = masterOrder.copy()
masterIndexSocket = range(len(masterIndexLabels))

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
configFile = 'datafolders.txt'


class STRASmaster:
    def __init__(self, host, port):
        """Initialize a STRAS slave port

        Args:
            host (str): IP address of the STRAS computer
            port (str): Port asigned to the slave
        """
        # Connect
        self.host = host
        self.port = port
        self.STRASmaster = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
        self.axisCurrVal = np.array([], dtype=np.float32)
        self.axisCurrValVel = np.array([], dtype=np.float32)
        self.axisCurrValNormalized = np.array([], dtype=np.float32)
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

        # For reference
        self.axisNames = masterOrder

    # --------------------------------------------------------------------------
    # ------------------------ Easy commands
    # --------------------------------------------------------------------------
    def easyStart(self):
        """Easy command to start the socket.
        """
        self.setTimeOut(1.0)
        self.connect(True)

    # --------------------------------------------------------------------------
    # ------------------------ Miscellaneous commands
    # --------------------------------------------------------------------------
    def setTimeOut(self, tx):
        """Sets timeout to the initialized STRAS slave node

        Args:
            tx (float): Number of seconds to wait on the socket
        """
        # ------------------------ No expected message reply
        self.STRASmaster.settimeout(float(tx))

    def connect(self, printOpt_=True):
        """Manually connects to the STRAS socket with the previously defined
        configuration

        Args:
            printOpt_ (bool, optional): To print outputs. Defaults to True.
        """
        # ------------------------ No expected message reply
        if self.STRASmaster.connect_ex((self.host, self.port)) == 0:
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
        self.STRASmaster.close()

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
            bytes
        """
        data = self.sendCMDmsg2STRAS(msgtype2send=1000,
                                     replyMode=1,
                                     xtraV='',
                                     printOpt=printOpt_)
        self.cycle = self.cycle + 1
        return data

    # ------------------------ Simulated Initialization
    def simulatedInit(self, printOpt_=False):
        """Simulate a connectection to the STRAS slave socket

        Args:
            printOpt_ (bool, optional): Verbose. Defaults to False.
        """
        self.NumberAxis = 12
        self.axisMins = [0, -163, -44, -1,
                         0, -163, -44, -1,
                         -1, -1, -1, -1]
        self.axisMaxs = [97, 163, 44, 1,
                         97, 163, 44, 1,
                         1, 1, 1, 1]
        self.ActiveAxis = [False]*self.NumberAxis
        if printOpt_:
            for i_ in range(0, self.NumberAxis):
                print(masterIndexLabels[i_], '', masterIndexSocket[i_], '\t',
                      ('ACTIVE' if self.ActiveAxis[i_] > 0 else 'INACTIVE'))
                print('\tmin: ', self.axisMins[i_])
                print('\tmax: ', self.axisMaxs[i_])

    # ------------------------ Simulated State
    def simulatedState(self, printOpt_=False, q_in_=0, q_in_norm=0):
        """Simulate a state update to the STRAS slave socket

        Args:
            printOpt_ (bool, optional): Verbose. Defaults to False.
            q_in_ (int, optional): List of values to be set on the fake socket.
            Defaults to 0.
            q_in_norm (int, optional): List of  normalized values to be set on
            the fake socket. Defaults to 0.
        """
        if q_in_ == 0 and q_in_norm == 0:
            q_simMaster = np.array([45, 0, -10, -1,
                                   45, 0, -10, -1,
                                   0, 0, 0, 0])
            q_simMasterNormalized = np.array([0.53, 0, 0.38, 0,
                                             0.53, 0, 0.38, 0,
                                             0, 0, 0, 0])
        else:
            q_simMaster = q_in_.copy()
            q_simMasterNormalized = q_in_norm.copy()
        self.axisCurrVal = q_simMaster.copy()
        self.axisCurrValNormalized = q_simMasterNormalized.copy()
        if printOpt_:
            print('Axis    \t AXmea  \tNorm')
            for i_ in range(0, self.NumberAxis):
                print(masterIndexLabels[i_], '\t',
                      masterIndexSocket[i_], '   \t',
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
        """
        Message locked to 33 bytes
        """
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
            self.STRASmaster.sendall(msg2stras)  # Send message
        except Exception:
            self.errorSendMsg = True
        try:
            data = self.STRASmaster.recvmsg(4096)
            return data
        except socket.timeout:
            return 'err:timeout'
        except Exception:
            self.errorRecvMsg = True
            print('Couldnt receive data! - INCOMPATIBLE MESSAGE')
            return 0

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
                print('Synchro OK ...')
                if printOpt:
                    print('INCOMING Message type',
                          messageTypeDic[r_STRAS_type],
                          'with len\t', r_STRAS_len-24-1, 'bytes')
            elif r_STRAS_syn != '!**!' and self.cycle == 0:
                print('Check Synchro!')
            if expectedReply == 71:
                STRASmaster.msgCMD_AXIS_infos(self, r_STRAS, printOpt)
            if expectedReply == 7:
                STRASmaster.msgCMD_RECV_STATE(self, r_STRAS, printOpt)
            if expectedReply == 410:
                STRASmaster.msgCMD_RECV_STATE(self, r_STRAS, printOpt)
            # if expectedReply == 3:
            #     STRASmaster.msgCMD_RECV_STATE(self, r_STRAS, printOpt)

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
                print(self.axisNames[i], i+1, '\t',
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
            print('Axis    \t AXmea  \t AXref  \t ME-REF')
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
            self.axisButtons = []
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

            # Inverting the Translation R, L
            if (i == 4) or (i == 0) or (i == 1) or (i == 5):
                self.axisCurrValNormalized = np.append(
                    self.axisCurrValNormalized,
                    np.interp(ax_meas[0],
                              (self.axisMins[i],
                              self.axisMaxs[i]), (1, 0)))
                # self.axisCurrValNormalized = np.append(
                #     self.axisCurrValNormalized, 0)
            # elif (i == 1) or (i == 5):
            #     self.axisCurrValNormalized = np.append(
            #         self.axisCurrValNormalized,
            #         np.interp(ax_meas[0],
            #                   (self.axisMins[i],
            #                    self.axisMaxs[i]), (1, 0)))
            else:
                self.axisCurrValNormalized = np.append(
                    self.axisCurrValNormalized,
                    np.interp(ax_meas[0],
                              (self.axisMins[i],
                              self.axisMaxs[i]), (0, 1)))

            if i == self.NumberAxis-1:
                last_ix = ix_1+3+8+8+8
            if printOpt:
                print(masterOrder[i], '', i+1, '   \t',
                      np.round(ax_meas[0], 4),
                      '   \t', np.round(ax_ref[0], 2),
                      '   \t', np.round(ax_meas[0]-ax_ref[0], 2))

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
        if type(r_STRAS) == tuple and len(r_STRAS) != 0:
            self.decodeMsg(r_STRAS, expectedReply=7, printOpt=printOpt_)
            if self.port == 6667:
                for i in range(len(self.axisCurrVal)):
                    if i in range(7):
                        # To update min and max JUST IN CASE - CALIBRATION
                        if self.axisCurrVal[i] > self.axisMaxs[i]:
                            self.axisMaxs[i] = self.axisCurrVal[i]
                        if self.axisCurrVal[i] < self.axisMins[i]:
                            self.axisMins[i] = self.axisCurrVal[i]

                    # # To normalize the current value
                    # self.axisCurrValNormalized[i] = (
                    #     (self.axisCurrVal[i] - self.axisMins[i]
                    #      )/(self.axisMaxs[i] - self.axisMins[i]))
                self.cycle = self.cycle + 1
        elif r_STRAS == 'err:timeout':
            print('Timeout error in Master ...')
        else:
            print('DATA RECEIVED NOT USEFUL!\n', r_STRAS)

        # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # ------------------------ For visualization
    # --------------------------------------------------------------------------
    def visualizeState(self, printOpt_=False):
        """Function to conver the state on master range and order to the control
        space.

        Args:
            printOpt_ (bool, optional): Verbose. Defaults to False.

        Returns:
            np.array: a numpy array with the rendered robot in the camera image
            plane
        """
        img = self.visualizeFrame(
            map_norm2range(
                order_slave2qp(self.axisCurrValNormalized),
                orderRanges(qlims_down_joint_pos, qpOrder),
                orderRanges(qlims_up_joint_pos, qpOrder)))
        return img

    def visualizeFrame(self, q2visualize, printOpt_=False):
        """Visualize the frame corresponding to q2visualize list-or-np.array

        Args:
            q2visualize (list): List of DOF of the STRAS robot in qp form and
            in qp range.
            printOpt_ (bool, optional): Verbose. Defaults to False.

        Returns:
            np.array: a numpy array with the rendered robot in the camera image
            plane
        """
        if not(self.initializeRenderFlag):
            self.initializeVisualization(self, False)
        strasOutput_slave = self.robot_stras.computemodels(
            q2visualize.copy())
        Tr_e_slave = strasOutput_slave[0]
        # Jr_e_slave = strasOutput_slave[1]
        Tl_e_slave = strasOutput_slave[2]
        # Jl_e_slave = strasOutput_slave[3]
        Tr_0_slave = strasOutput_slave[4]
        # Jr_0_slave = strasOutput_slave[5]
        Tl_0_slave = strasOutput_slave[6]
        # Jl_0_slave = strasOutput_slave[7]
        Te_slave = strasOutput_slave[8]
        # Je_slave = strasOutput_slave[9]
        shpRslve, shapeLslave, tipend_slave = self.robot_stras.computeShape(
            q2visualize.copy())
        img_ = self.render.render_stras_arms(
            shpRslve, shapeLslave, tipend_slave, Tr_e_slave,
            Tl_e_slave, Tr_0_slave, Tl_0_slave, Te_slave)
        return img_

    def initializeVisualization(self, printOpt_=False):
        """To initialize the visualization renders and else.

        Args:
            printOpt_ (bool, optional): Verbose. Defaults to False.
        """
        d = {}
        with open(configFile) as f:
            for line in f:
                (key, val) = line.split("=")
                d[key] = val

        # Create camera and robot model
        cam = camera_model(
            d['camera_calibration'][:-1])
        robot_right = robot_model(
            configfile=d['robot_config_right'][:-1])
        robot_left = robot_model(
            configfile=d['robot_config_left'][:-1])
        robot_end = robot_twobending(
            configfile=d['robot_config_endoscope'][:-1])

        # Create the overall stras robot model
        # (to compute pose and jacobian of each robot)
        self.robot_stras = StrasModel(robot_right, robot_left, robot_end)

        # Create rendering to render robot shape
        w, h = cam.getImSize()
        self.render = render_shape(w, h, self.robot_stras)
        self.render.setCamProperties(cam.getF(), cam.getC(), cam.K)

        # Update the initialization flag
        self.initializeRenderFlag = True
