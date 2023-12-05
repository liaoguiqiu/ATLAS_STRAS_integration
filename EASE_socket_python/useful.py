# import numpy as np
# from future.utils import is_new_style
# from sympy.geometry import Point
# from sympy.geometry import Line
# from sympy import Float
from math import tan, radians
import numpy as np
from math import isclose
# from liegroups import SE3

slaveOrder_local = ['EBV', 'EBH', 'ER', 'ET',
                    'TR', 'RR', 'BR',
                    'TL', 'RL', 'BL']
tolIsCloseSlave_local = [0.5, 0.5, 2, 1,
                         2, 2, 0.2,
                         2, 2, 0.2]
tooCloseDict_local = dict(zip(slaveOrder_local, tolIsCloseSlave_local))
tooIsCloseSlave_qp = [0.0005, 0.0005, 1, 0.2,
                      0.0005, 1, 0.2,
                      0.0005, 1, 0.2]
tooCloseDict_qp = dict(zip(slaveOrder_local, tooIsCloseSlave_qp))
maxRelativeDict = dict(zip(slaveOrder_local,
                           [0.1, 0.1, 0.1, 0.1,
                            0.1, 0.1, 0.1,
                            0.1, 0.1, 0.1]))
slaveLabels_teleoperate_local = ['EBV', 'EBH', 'ET', 'ER']
slaveOrder_teleoperate_local = [0.25, 0.25, 1, 1]
stepEndoTeleoperation = dict(
    zip(slaveLabels_teleoperate_local, slaveOrder_teleoperate_local))
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
qlims_top_slave_pos = {'BR': 6.5,  # deg/s
                       'RR': 360,  # Rotation in degrees/s
                       'TR': 73.5,  # Translation in mm/s
                       'BL': 6.5,  # deg/s
                       'RL': 360,  # Rotation in degrees/s
                       'TL': 73.5,  # Translation in mm/s
                       'EBH': 9.12,  # HorB-E
                       'EBV': 9.12,  # VertB-E
                       'ER': 50,  # Rotation ENDO in degrees/s
                       'ET': 90}  # Translation ENDO in mm/s
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
qlims_top_joint_pos = {'BR': np.pi/(2*18.5),  # deg/s
                       'RR': radians(180),  # Rotation in degrees/s
                       'TR': 75,  # Translation in mm/s
                       'BL': np.pi/(2*18.5),  # deg/s
                       'RL': radians(180),  # Rotation in degrees/s
                       'TL': 75,  # Translation in mm/s
                       'EBH': np.pi/(2*185),  # HorBend-E
                       'EBV': np.pi/(2*185),   # VertBnd_E
                       'ER': radians(358),  # Rotation ENDO in degrees/s
                       'ET': 100}  # Translation ENDO in mm/s
lims_down = np.array(list(qlims_down_joint_pos.values()))
lims_up = np.array(list(qlims_top_joint_pos.values()))
slaveOrder = ['EBV', 'EBH', 'ER', 'ET',
              'TR', 'RR', 'BR',
              'TL', 'RL', 'BL']
masterOrder = ['TL', 'RL', 'BL', 'AL',
               'TR', 'RR', 'BR', 'AR',
               'EBV', 'EBH', 'ET', 'ER']
qpOrder = ['BR', 'RR', 'TR',
           'BL', 'RL', 'TL',
           'EBV', 'EBH', 'ER', 'ET']


# To generate random q
def generate_random_q():
    # curvature between -0.085 and 0.085 mm -1
    k_r = (0.085)*np.random.random_sample()
    k_l = (0.085)*np.random.random_sample()

    # angle between -90 and 90
    a_r = (2.0*np.pi/2.0)*np.random.random_sample() + np.pi/2.0
    a_l = (2.0*np.pi/2.0)*np.random.random_sample() - np.pi/2.0

    # insertion between 0 and 60
    d_r = 60*np.random.random_sample()
    d_l = 60*np.random.random_sample()

    # # endoscope steady for testing
    k_e = (0.005)*np.random.random_sample()
    a_e = 0.0
    d_e = 10*np.random.random_sample()
    #
    # # test values
    # k_r = 0.085 * np.random.random_sample()  #
    # k_l = 0.085 * np.random.random_sample()  #
    # a_r = np.pi
    # a_l = 0.0
    return [k_r, a_r, d_r, k_l, a_l, d_l, k_e, a_e, d_e]


# Filter q2cmd w.r.t. q_dot>0 values
def filter_q2cmd(q_in_, labels_in_):
    """Filter q_in with respect to the labels_in

    Args:
        q_in_ (list): Joint values
        labels_in_ (list): Labels of the joint values

    Returns:
        list: list of list of commands to the STRAS slave
    """
    q2cmd_out_ = []
    for u_ in range(len(labels_in_)):
        idx_ = [k_ for k_ in range(
            len(qpOrder)) if labels_in_[u_] == slaveOrder[k_]][0]
        q2cmd_out_.append([labels_in_[u_],
                           q_in_[idx_],
                           maxRelativeDict[labels_in_[u_]]])
    return q2cmd_out_


# To check which control signal to update
def filter_q_dot(q_In_, q_Dot_, onlyDof_=None):
    """Filter the intented q_dot for the socket w.r.t. to the onlyDof list
    and the dictionary of minimum step tooCloseDict_up

    Args:
        q_In_ (list): Joint values
        q_Dot_ (list): Q dot of joint values
        onlyDof_ (list, optional): To filter to selected DOF. Defaults to None.

    Returns:
        list: list of list to send to the STRAS slave
    """
    cmd2Stras_internal = []
    dict1_ = dict(zip(slaveOrder_local, q_In_))
    dict2_ = dict(zip(slaveOrder_local, q_Dot_))
    if onlyDof_ is not None:
        dof2Use_ = onlyDof_.copy()
    else:
        dof2Use_ = slaveOrder_local.copy()
    vals1_ = [dict1_[x_] for x_ in dof2Use_]
    vals2_ = [dict2_[x_] for x_ in dof2Use_]
    for i_ in range(len(dof2Use_)):
        if not(isclose(vals1_[i_], vals2_[i_],
               abs_tol=tooCloseDict_qp[dof2Use_[i_]])):
            cmd2Stras_internal.append([dof2Use_[i_],
                                       q_Dot_[i_],
                                       maxRelativeDict[dof2Use_[i_]]])
    return cmd2Stras_internal


# From q2tele and raw q_slave activate the DOF that are close (master-slave)
def defineActiveDof(q2Tele_, q_slave_, dof2use, dictActivation_, pOpt_=False):
    """Reactivate the DOF that are close to each other

    Args:
        q2Tele_ (list): Joint positions to send
        q_slave_ (list): Joint positions from the Endoscope slave
        dof2use (list): List of DOF to control
        dictActivation_ (dict): Dictionary of active DOF [True or False]
        pOpt_ (bool, optional): To print verbose. Defaults to False.

    Returns:
        dict: Dictionary of DOF to control
    """
    dictSlave = dict(zip(slaveOrder_local, q_slave_))
    dictq2Tel = dict(zip(slaveOrder_local, q2Tele_))
    valsSlave = [dictSlave[x_] for x_ in dof2use]
    valsTele = [dictq2Tel[x_] for x_ in dof2use]
    for i_ in range(len(dof2use)):
        if not(dictActivation_[dof2use[i_]]):
            if isclose(valsSlave[i_], valsTele[i_],
                       abs_tol=tooCloseDict_local[dof2use[i_]]):
                dictActivation_[dof2use[i_]] = True
                if pOpt_:
                    print('Retaking control of: ', dof2use[i_])
    return dictActivation_


# From the q2tele and the activation dictionary generate the commands
def teleoperateSTRAS(q2Tele_, dictActivation_local_):
    """From the q2Tele_ filter w.r.t. the activation_local_

    Args:
        q2Tele_ (list): Joint values to send to the STRAS slave
        dictActivation_local_ (dict): Dictionary with active or not DOF

    Returns:
        list: list of list to send to the STRAS slave
    """
    cmd2stras_ = []
    q2teleDict = dict(zip(slaveOrder_local, q2Tele_))
    list_ = [i_ for i_ in range(
        len(dictActivation_local_)) if list(
            dictActivation_local_.values())[i_]]
    if len(list_) >= 1:
        for idx_ in list_:
            # print('slaveOrder 2 control', slaveOrder[idx_])
            # print('value2stras', q2teleDict[slaveOrder[idx_]])
            # print('maxRelative', maxRelativeDict[slaveOrder[idx_]])
            cmd2stras_.append([slaveOrder_local[idx_],
                               q2teleDict[slaveOrder_local[idx_]],
                               maxRelativeDict[slaveOrder_local[idx_]]])
    return cmd2stras_


# To teleoperate endoscope from the [-1, 1] signals from the master joysticks
def teleoperateSTRASendo(qEndo_in_, qSlave_Endo_):
    """Create the teleoperation comand to send to the STRAS slave ENDOSCOPE dof

    Args:
        qEndo_in_ (list): List of DOF of endoscope joint positions to send
        qSlave_Endo_ (list): List of DOF of endoscope joint positions of slave

    Returns:
        list: [description]
    """
    qEndo_ = [qEndo_in_[0], qEndo_in_[1], qEndo_in_[3], qEndo_in_[2]]
    dofLabelList_ = ['EBV', 'EBH', 'ER', 'ET']
    qEndoOut_ = []
    # stepEndoTeleoperation
    for i_, dofEndo_ in enumerate(qEndo_):
        if int(dofEndo_) != 0:
            qEndoOut_.append(
                [dofLabelList_[i_],
                 qSlave_Endo_[i_]+(
                     stepEndoTeleoperation[dofLabelList_[i_]]*dofEndo_),
                 0.1])
    return qEndoOut_


# Order the values of a dictionary w.r.t. the orderListOuput
def orderRanges(rangeDict, orderListOutput):
    """Order a dictionary values in a list order

    Args:
        rangeDict (dict): Dictionary with dof labels as keys and determined
        value
        orderListOutput (list): List of the labels of the dictionary in a
        specified order

    Returns:
        np.array: [description]
    """
    outputList = []
    for lbl in orderListOutput:
        outputList.append(rangeDict.get(lbl))
    return np.array(outputList)


# To interpolate from [0, 1] to [mins, maxs] the Slave q
def map_norm2range(q_in_, listOfmins_, listOfmaxs_):
    """Normalize the input list w.r.t. to the desired mins and maxs

    Args:
        q_in_ (list): List of values to interpolate
        listOfmins_ (list): List of mins
        listOfmaxs_ (list): List of maxs

    Returns:
        list: Input in the correct range
    """
    # Input should be normalized and increasing between [0, 1]
    if type(listOfmins_) is np.ndarray:
        listOfmins_ = listOfmins_.tolist().copy()
    if type(listOfmaxs_) is np.ndarray:
        listOfmaxs_ = listOfmaxs_.tolist().copy()
    q_out_ = np.zeros(len(q_in_))
    for i_ in range(len(q_in_)):
        q_out_[i_] = np.interp(q_in_[i_],
                               (0, 1),
                               (listOfmins_[i_], listOfmaxs_[i_]))
    return q_out_


# To interpolate from [0, 1] to [mins, maxs] the QP q
def map_norm2slave_slaveOrder(q_in_, listOfmins_, listOfmaxs_):
    """Normalize the input list w.r.t. to the desired slave mins and maxs

    Args:
        q_in_ (list): List of values to interpolate
        listOfmins_ (list): List of mins
        listOfmaxs_ (list): List of maxs

    Returns:
        list: Input in the correct range
    """
    # Input should be normalized and increasing between [0, 1]
    if type(listOfmins_) is np.ndarray:
        listOfmins_ = listOfmins_.tolist().copy()
    if type(listOfmaxs_) is np.ndarray:
        listOfmaxs_ = listOfmaxs_.tolist().copy()
    q_out_ = np.zeros(len(q_in_))
    q_in_[4] = 1 - q_in_[4]  # Invert Right translation
    q_in_[7] = 1 - q_in_[7]  # Invert Left translation
    q_in_[9] = 1 - q_in_[9]  # Invert Left bending
    for i_ in range(len(q_in_)):
        # if i_ == 4 or i_ == 6 or i_ == 7:
        #     q_out_[i_] = np.interp(q_in_[i_],
        #                            (0, 1),
        #                            (listOfmaxs_[i_], listOfmins_[i_]))
        # else:
        q_out_[i_] = np.interp(
            q_in_[i_],
            (0, 1),
            (listOfmins_[i_], listOfmaxs_[i_]))
    return q_out_


# To check if the list is within the min and max, return list of errors
def checkInRange(list_, mins_, maxs_):
    """Check if a list is within the mins and maxs ranges

    Args:
        list_ (list): List of values
        mins_ (list): List of mins
        maxs_ (list): List of mins

    Returns:
        list: result, flag
    """
    res = []
    flag_ = False
    for i in range(len(list_)):
        a = []
        if list_[i] < mins_[i]:
            a.append('min')
            flag_ = True
        if list_[i] > maxs_[i]:
            a.append('max')
            flag_ = True
        res.append(a)
    return res, flag_


# To normalize q (range:lim_down and lim_up) to [0,1], SAME ORDERS!!!
def normalize_q_qp(q_qp, lim_down_, lim_up_):
    """Normalize q (range:lim_down and lim_up) to [0,1], SAME ORDERS!!!

    Args:
        q_qp (list): Joint positions
        lim_down_ (list): List of mins
        lim_up_ (list): List of maxs

    Returns:
        list: list of range range [0, 1]
    """
    q_out = np.zeros(len(q_qp))
    for i in range(len(q_qp)):
        q_out[i] = np.interp(q_qp[i],
                             (lim_down_[i], lim_up_[i]),
                             (0, 1))
    return q_out


# Change order from qp to slave
def order_qp2slave(q_in_):
    """Order the list(q_in_) from the qp order to the slave

    Args:
        q_in_ (list): Joint values in the corrected order

    Returns:
        list: List in the correct order
    """
    q_slave_endo = [q_in_[6], q_in_[7], q_in_[8], q_in_[9]]
    q_slave_R = [q_in_[2], q_in_[1], q_in_[0]]
    q_slave_L = [q_in_[5], q_in_[4], q_in_[3]]
    q_out_ = q_slave_endo + q_slave_R + q_slave_L
    return q_out_


# Change order from master to qp
def order_master2qp(q_in_):
    """Order the list(q_in_) from the master order to the qp

    Args:
        q_in_ (list): Joint values in the corrected order

    Returns:
        list: List in the correct order
    """
    q_R = [q_in_[6], q_in_[5], q_in_[4]]
    q_L = [q_in_[2], q_in_[1], q_in_[0]]
    q_e = [q_in_[8], q_in_[9], q_in_[11], q_in_[10]]  # CHECK
    q_out_ = q_R + q_L + q_e
    return q_out_


# Change order from slave to qp
def order_slave2qp(q_in_):
    """Order the list(q_in_) from the master order to the slave

    Args:
        q_in_ (list): Joint values in the corrected order

    Returns:
        list: List in the correct order
    """
    q_R = [q_in_[6], q_in_[5], q_in_[4]]
    q_L = [q_in_[9], q_in_[8], q_in_[7]]
    q_e = [q_in_[0], q_in_[1], q_in_[2], q_in_[3]]
    q_out_ = q_R + q_L + q_e
    return q_out_


# Mapping functions between normalized q's
def order_master2slave(q_in_):
    """Order the list(q_in_) from the master order to the slave

    Args:
        q_in_ (list): Joint values in the corrected order

    Returns:
        list: List in the correct order
    """
    q_E = [q_in_[8], q_in_[9], q_in_[11], q_in_[10]]
    q_R = [q_in_[4], q_in_[5], q_in_[6]]
    q_L = [q_in_[0], q_in_[1], q_in_[2]]
    q_out_ = q_E + q_R + q_L
    return q_out_


# Funtion to obtain if point is below or above the FOV line and the distance
def distance_point_line(p1, pl_1, pl_2, inwards_d, arm):
    """Obtain if point is below or above the FOV line and the distance

    Args:
        p1 (np.array): point of the tip of the instrument
        pl_1 (np.array): init point of the FOV segment to consider
        pl_2 (np.array): final point of the FOV segment to consider
        inwards_d (np.array): Normal distance inwards the FOV lines
        arm (str): Type of arm 'r','R' or 'l','L'

    Returns:
        list: 1-signed distance to the FOV lines
              2-position of the projected line over the FOV lines times inward
    """
    d = np.linalg.norm(np.cross(pl_2-pl_1, p1-pl_1))/np.linalg.norm(pl_2-pl_1)
    pos, _ = projection_over_line(pl_1, pl_2, p1, inwards_d, arm)
    if (p1-pos)[-1] < 0:
        d_symbol = -1
    else:
        d_symbol = 1
    return d*d_symbol, pos


# Funtion to obtain the projection to the closest FOV point
def projection_over_line(a, b, p, inwards_d, arm):
    """Obtain the projection to the closest FOV point

    Args:
        a (np.array): init point of the FOV segment to consider
        b (np.array): final point of the FOV segment to consider
        p (np.array): point of the tip of the instrument
        inwards_d ([type]): Normal distance inwards the FOV lines
        arm ([type]): String of the arm 'r','R' or 'l','L'

    Returns:
        list: 1-projection by d(inwards) over the FOV lines
              2-point online closest to p
    """
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    # if you need the the closest point belonging to the segment
    t = max(0, min(1, t))
    result = a + t * ab
    result[1] = result[1] + inwards_d
    res_online = result
    if arm == 'R' or arm == 'r':
        result[0] = result[0] - inwards_d
    elif arm == 'L' or arm == 'l':
        result[0] = result[0] + inwards_d
    elif arm == 'U' or arm == 'u':
        result[0] = result[0] + inwards_d
    elif arm == 'D' or arm == 'd':
        result[0] = result[0] - inwards_d
    return result, res_online


# # Function to get the distance to the FOV pyramid - Horizontal
# def dist2PovHorizontal(point, quadrant,
#                        angle=50, distanceInit=10, distanceEnd=80):
#     """Function to get the distance to the FOV pyramid - Horizontal

#     Args:
#         point (list): List of current point location
#         quadrant (str): Quadrant on the top or bottom
#         angle (int, optional): Angle of vertical FOV. Defaults to 40.
#         distanceInit (int, optional): Starting distance of the line that builds
#         the FOV pyramid. Defaults to 20.
#         distanceEnd (int, optional): End distance of the line that builds
#         the FOV pyramid. Defaults to 50.

#     Returns:
#         [type]: [description]
#     """
#     posPointInit = Float(
#         Float(tan(radians(angle)), 3)*distanceInit, 2)
#     posPointEnd = Float(Float(tan(radians(angle)), 3)*distanceEnd, 2)
#     # ------------------------ Defining the points of the boundary POV
#     #     p0 = Point(0,0,0)
#     pLeftInit = Point(-posPointInit, 0, distanceInit)
#     pRightInit = Point(posPointInit, 0, distanceInit)
#     pLeftEnd = Point(-posPointEnd, 0, distanceEnd)
#     pRightEnd = Point(posPointEnd, 0, distanceEnd)
#     # ------------------------ Shortest distance to POV boundary line
#     povLeft = Line(pLeftInit, pLeftEnd)
#     povRight = Line(pRightInit, pRightEnd)
#     pActual = Point(point)
#     if quadrant == 'R':
#         return round(povRight.distance(pActual), 3)
#     elif quadrant == 'L':
#         return round(povLeft.distance(pActual), 3)
#     else:
#         print('Specify quadrant')
#         return 'Error'


# # Function to get the distance to the FOV pyramid - Vertical
# def dist2PovVertical(point, quadrant,
#                      angle=40, distanceInit=20, distanceEnd=50):
#     """Function to get the distance to the FOV pyramid - Vertical

#     Args:
#         point (list): List of current point location
#         quadrant (str): Quadrant on the top or bottom
#         angle (int, optional): Angle of vertical FOV. Defaults to 40.
#         distanceInit (int, optional): Starting distance of the line that builds
#         the FOV pyramid. Defaults to 20.
#         distanceEnd (int, optional): End distance of the line that builds
#         the FOV pyramid. Defaults to 50.

#     Returns:
#         [type]: [description]
#     """
#     posPointInit = Float(
#         Float(tan(radians(angle)), 3)*distanceInit, 2)
#     posPointEnd = Float(Float(tan(radians(angle)), 3)*distanceEnd, 2)
#     # ------------------------ Defining the points of the boundary POV
#     #     p0 = Point(0,0,0)
#     pLeftInit = Point(-posPointInit, 0, distanceInit)
#     pRightInit = Point(posPointInit, 0, distanceInit)
#     pLeftEnd = Point(-posPointEnd, 0, distanceEnd)
#     pRightEnd = Point(posPointEnd, 0, distanceEnd)
#     # ------------------------ Shortest distance to POV boundary line
#     povLeft = Line(pLeftInit, pLeftEnd)
#     povRight = Line(pRightInit, pRightEnd)
#     pActual = Point(point)
#     if quadrant == 'U':
#         return round(povRight.distance(pActual), 3)
#     elif quadrant == 'D':
#         return round(povLeft.distance(pActual), 3)
#     else:
#         print('Specify quadrant')
#         return 'Error'

# Function to obtain distance and direction between points
def consPoints(p1_, p2_):
    """Function to obtain distance and direction between points

    Args:
        p1_ (list): List of point 1
        p2_ (list): List of point 2

    Returns:
        list: distance between the points, direction
    """
    p1 = np.array(p1_)
    p2 = np.array(p2_)
    distanceBetweenPoints = np.linalg.norm(p1-p2)
    directionToP2fromP1 = (p2-p1)/distanceBetweenPoints
    return distanceBetweenPoints, directionToP2fromP1
