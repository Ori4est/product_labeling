from __future__ import print_function
from __future__ import division
import glob
import os
import cv2 
from math import atan2, cos, sin, sqrt, pi
import numpy as np
from sympy.geometry import Line, Point


def extend_line(p1, p2, distance=10000):
    p1_x = int(p1[0])
    p1_y = int(p1[1])
    p2_x = int(p2[0])
    p2_y = int(p2[1])
    diff = np.arctan2(p1_y - p2_y, p1_x - p2_x)
    p3_x = int(p1[0] + distance*np.cos(diff))
    p3_y = int(p1[1] + distance*np.sin(diff))
    p4_x = int(p1[0] - distance*np.cos(diff))
    p4_y = int(p1[1] - distance*np.sin(diff))
    return ((p3_x, p3_y), (p4_x, p4_y))


def draw_axis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)


def get_orient(pts, show_img):
    '''performs PCA & get orientation
    pts: contour of interst
    img: corresponding input image
    src: https://github.com/opencv/opencv/blob/4.x/samples/python/tutorial_code/ml/introduction_to_pca/introduction_to_pca.py
    '''
    sz = len(pts) 
    data_pts = np.empty((sz, 2), dtype=np.float64) 
    for i in range(data_pts.shape[0]): 
        data_pts[i, 0] = pts[i, 0, 0] 
        data_pts[i, 1] = pts[i, 0, 1] 

    # Perform PCA analysis 
    mean = np.empty((0)) 
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object 
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    cv2.circle(show_img, cntr, 3, (255, 0, 255), 2) # peachy pink
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0]) 
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0]) 
    draw_axis(show_img, cntr, p1, (0, 150, 0), 1) # green
    draw_axis(show_img, cntr, p2, (200, 150, 0), 5) # mustar

    # draw reference axis 
    hor = (int((cntr[0] - show_img.shape[0]/2)), cntr[1])
    ver = (cntr[0], int(cntr[1] + show_img.shape[1]/2))
    draw_axis(show_img, cntr, hor, (64, 224, 208), 1) # turquoise
    draw_axis(show_img, cntr, ver, (105, 105, 105), 1) # dimgray
    #custom_angle = atan2(p1[0]-ver[0], p1[1]-ver[1])
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0]) # orientation in radians
    if p1[1] > cntr[1]:
        angle_deg = 90 - int(np.rad2deg(angle))
    else:
        angle_deg = 90 - int(np.rad2deg(angle)) - 180
    # Label with the rotation angle [str(-int(np.rad2deg(angle)) - 90)] 
    label = "  Rotation Angle: " + str(angle_deg) + " deg (eign)"
    textbox = cv2.rectangle(show_img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1) # white
    cv2.putText(show_img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    print("comparing p1, cntr, angle", p1, cntr, angle, angle_deg)
    return angle_deg, cntr, (p1, p2)


def GetRotateMatrixWithCenter(x, y, angle):
    # https://math.stackexchange.com/questions/2093314
    """ x: img_width // 2 | aka mean_x
        y: img_height // 2 | aka mean_y
        angle: rotation in deg
    """
    angle_rad = np.deg2rad(angle) # degree to radian
    move_matrix = np.array(
        [
            [1, 0, x], 
            [0, 1, y], 
            [0, 0, 1]
        ])
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0], 
            [np.sin(angle),  np.cos(angle), 0], 
            [0,             0,              1]
        ])
    back_matrix = np.array(
        [
            [1, 0, -x], 
            [0, 1, -y], 
            [0, 0, 1]
        ])

    r = np.dot(move_matrix, rotation_matrix)
    return np.dot(r, back_matrix)



def img_rotate(image, angle):
    size_reverse = np.array(image.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.) # rotation around its centre
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(image, M, tuple(size_new.astype(int)))


def img_transform(roi_shape, roi_cntr, roi_angle, dst_img_shape, overlay_img):
    # getting affine transformation form sample image to template
    size_reverse = np.array(dst_img_shape[::-1])
    # given roi_cntr, rota_angle & scale
    rows, cols = overlay_img.shape[:2]
    overlay_pts = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])
    fix_scale = min(roi_shape[0]/rows, roi_shape[1]/cols)
    fix_img_shape = tuple([int(rows * fix_scale), int(cols * fix_scale)])
    roi_pts = cv2.boxPoints((roi_cntr, (fix_img_shape), roi_angle))
    M = cv2.getPerspectiveTransform(overlay_pts, roi_pts)

    return cv2.warpPerspective(overlay_img, M, size_reverse)


def create_contour(prod_img):
    
    # transparent to black
    if prod_img.shape == 4:
        prod_img[np.where(prod_img[:, :, 3] == 0)] = (0, 0, 0, 255)
        fill_img = prod_img[:, :, 0:3]
    else:
        fill_img = prod_img
    fill_img[np.where(fill_img[:, :, 0] < 60)] = (0, 0, 0)
    fill_img[np.where(fill_img[:, :, 1] < 60)] = (0, 0, 0)
    fill_img[np.where(fill_img[:, :, 2] < 60)] = (0, 0, 0)
    gray_img = cv2.cvtColor(fill_img, cv2.COLOR_BGR2GRAY)
    _, bw_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # no hierarchy
    # sort contours by largest first 
    max_area = -1
    best_cnt = None

    for cnt in contours:
    
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    return best_cnt


def create_roi(contour_mean, axis, img):
    """
    cnt_pts: set of contour points
    img: cali_prod_img
    """
    p1, p2 = axis[0], axis[1]
    #moments = cv2.moments(cnt_pts)
    #hu = cv2.HuMoments(moments) # this is to find matching shape
    
    #roi_centre = (int(moments['m10']/moments['m00']), int(moments['m01'] / moments['m00']))
    #cv2.circle(img, roi_centre, 3, (191, 226, 159), -1)
    #cv2.drawContours(img, cnt_pts, 0, (255, 204, 204), 2)
    h, w = img.shape[:2]
    # bw_img as input_src of distTrans
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #_, bw_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # threshold to binary for floodfill
    # create zeros mask 2 pixels larger in each dim
    mask_img = np.zeros([h+2, w+2], np.uint8)
    # floodfill white between 
    ff_img = img.copy()
    ff_img = cv2.floodFill(ff_img, mask_img, (240, 240), 255)[1]
    cv2.imshow("floodfil before", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if ff_img[0][0] == 255: # TODO temporal solution
        ff_img = cv2.bitwise_not(ff_img)
    cv2.imshow("floodfill after", ff_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find output
    # find roi area
    contours, _ = cv2.findContours(ff_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # no hierarchy
    # sort contours by largest first 
    max_area = -1
    best_cnt = None

    for cnt in contours:
    
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    
    pca_angle, pca_cntr, _ = get_orient(best_cnt, ff_img)
    cv2.imshow("get orient", ff_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find the regions share the largest dist

    dist_transform = cv2.distanceTransform(ff_img, cv2.DIST_L2, 5)
    maxDT = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
    # axis line placement
    axis_line = Line(contour_mean, p2)
    parral_line = axis_line.parallel_line(max_loc)
    parral_ln_pts = np.asarray(parral_line.points)
    extent_ln_pts = extend_line(parral_ln_pts[0], parral_ln_pts[1])

    print("using distTrans", maxDT, max_val, max_loc)
    
    #cv2.circle(ff_img, (maxDT[1], maxDT[0]), int(max_val), (0, 255, 255), 2) # (img, center_coord, rad, color, thickness)
    cv2.line(ff_img, extent_ln_pts[0], extent_ln_pts[1], (0, 0, 100), 3, -1)
    
    cv2.drawContours(ff_img, [best_cnt], -1, (0, 0, 0), -1)
    cv2.imshow("line cut", ff_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #pca_cntr, pca_angle = get_mean(best_cnt)
    rect_cntr, rect_shape, rect_angle = cv2.minAreaRect(best_cnt) # (center(x, y), (width, height), angle of rotation)
    #rect_width, rect_height = minrect_helper(rect_shape, rect_angle)
    #print("True mean", pca_cntr, rect_cntr) # very close

    # resampled contour - set of (x, y) pts in shape (4, 1, 2) where 1 is the singleton dim
    #approx = cv2.approxPolyDP(cnt_pts, 0.01 * cv2.arcLength(cnt_pts, True), True)
    # find corner (x_3, y_3) (farest point ~ (x, y) pair with max product)
    #far = approx[np.prod(approx, 2).argmax()][0]
    # find mean (perform PCA) 
    #approx_mean = get_mean(approx)
    # strong assumption hold? distance between [near -- mean -- far] are equl
    #dist = far - approx_mean[0]
    #near = approx_mean[0] - dist
    #print(far, approx_mean, near)

    #ymax = approx[approx[:, :, 0] == near[0]].max()
    # corner (x_1, 1)
    #xmax = approx[approx[:, :, 1] == near[1]].max()

    # find min in (far.x, xmax) & (far.y, ymax)
    #x = min(far[0], xmax)
    #y = min(far[1], ymax)

    #roi = [near, far, approx_mean]
    #print(roi)
    # draw roi box inside the contour
    #cv2.rectangle(img, (int(near[0]), int(near[1])), (int(far[0]), int(far[1])), (0, 0, 255), 2)

    return pca_cntr, rect_shape, pca_angle


def minrect_helper(rota_shape, rota_angle):
    """
    inspired by https://stackoverflow.com/questions/69074165/order-of-corners-in-a-rotating-rectangle-in-opencv-python
    """
    rota_width, rota_height = rota_shape[0], rota_shape[1]
    if rota_height > rota_width:
        new_angle = - rota_angle
    else:
        new_angle = 90 - rota_angle

    if new_angle >= 0 and new_angle <= 90:
        new_width = rota_width
        new_height = rota_height
    elif new_angle > 90 and new_angle < 180:
        new_width = rota_height
        new_height = rota_width
    else:
        new_width = rota_height
        new_height = rota_width 
    
    return new_width, new_height


def get_mean(pts):
    sz = len(pts) 
    general_mean = np.empty((0))
    data_pts = np.empty((sz, 2), dtype=np.float64) 
    for i in range(data_pts.shape[0]): 
        data_pts[i, 0] = pts[i, 0, 0] 
        data_pts[i, 1] = pts[i, 0, 1]
    general_mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, general_mean)
    cntr = (int(general_mean[0, 0]), int(general_mean[0, 1]))
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    cntr = (int(general_mean[0, 0]), int(general_mean[0, 1]))
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0]) # orientation in radians
    if p1[1] > cntr[1]:
        angle_deg = 90 - int(np.rad2deg(angle))
    else:
        angle_deg = 90 - int(np.rad2deg(angle)) - 180
    return cntr, angle_deg


def create_rota(image_dir, angle_list, save_dir='./data/tmp/'):
    image_names = glob.glob(image_dir+"*.png")
    for image_name in image_names:
        if "Background" in image_name:
            continue
        for angle in angle_list:
            image = cv2.imread(image_name, -1)
            size_reverse = np.array(image.shape[1::-1]) # swap x with y
            M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
            MM = np.absolute(M[:, :2])
            size_new = MM @ size_reverse
            M[:, -1] += (size_new - size_reverse) / 2.
            rota_image = cv2.warpAffine(image, M, tuple(size_new.astype(int)))
            cv2.imwrite(os.path.join(save_dir, "{}_rota_{}.png".format(os.path.basename(image_name).split('.')[0], angle)), rota_image)


def get_edge_detection_thresholds(img):
    """Calculates the lower and upper thresholds for Canny edge detection"""
    sigma = 0.3
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return (lower, upper)


def draw_linesP(dst_img, linesP):
    if linesP is not None:
        for i in range(len(linesP)):
            rho = linesP[i][0][0]
            theta = linesP[i][0][1]
            a = cos(theta)
            b = sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(dst_img, pt1, pt2, (225,225,255), 3, cv2.LINE_AA)
