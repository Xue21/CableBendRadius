import cv2
import numpy as np
from matplotlib import pyplot as plt
from pyrealsense2 import pyrealsense2 as rs
rs2_distortion = rs.distortion
import pickle
import math
import time
from PIL import Image
from net.deeplab import DeeplabV3


def contains_point(points, x, y):
    p = np.array([x, y])
    return np.any(np.all(points == p, axis=1))  # 使用np.any判断np.all(points == p, axis=1)中是否有元素为True

def transitions(neighbours):
    """
    P2-P9-P2中从0到1以及从1到0的个数
    """
    return sum([neighbours[i] != neighbours[i + 1] for i in range(len(neighbours) - 1)])

def rs2_deproject_pixel_to_point(intrinsics, pixel, depth):
    # Get the focal length in x and y directions
    fx, fy = intrinsics.fx, intrinsics.fy
    # Get the principal point coordinates
    ppx, ppy = intrinsics.ppx, intrinsics.ppy
    # # Get the distortion coefficients
    # coeffs = intrinsics.coeffs

    # Calculate the normalized coordinates
    x = (pixel[0] - ppx) / fx
    y = (pixel[1] - ppy) / fy

    # if intrinsics.model == rs2_distortion.RS2_DISTORTION_INVERSE_BROWN_CONRADY:
    #     # Apply inverse Brown-Conrady distortion model
    #     r2 = x**2 + y**2
    #     f = 1 + coeffs[0] * r2 + coeffs[1] * r2**2 + coeffs[4] * r2**3
    #     ux = x * f + 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x**2)
    #     uy = y * f + 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 * y**2)
    #     x = ux
    #     y = uy
    # elif intrinsics.model == rs2_distortion.RS2_DISTORTION_BROWN_CONRADY:
    #     # Apply Brown-Conrady distortion model
    #     r2 = x**2 + y**2
    #     f = 1 + coeffs[0] * r2 + coeffs[1] * r2**2 + coeffs[4] * r2**3
    #     x *= f
    #     y *= f
    #     dx = x + 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x**2)
    #     dy = y + 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 * y**2)
    #     x = dx
    #     y = dy

    # Calculate the 3D coordinates in camera frame
    point3d = [depth * x, depth * y, depth]
    return point3d


def circle_radius(A, B, C):
    """计算外接圆半径"""
    v1 = np.array(B) - np.array(A)
    v2 = np.array(C) - np.array(A)
    n = np.cross(v1, v2)  # 向量叉乘

    S = 0.5 * np.linalg.norm(n)
    if S == 0:
        # 无法形成圆
        return None
    BC = np.linalg.norm(np.array(B) - np.array(C))
    AB = np.linalg.norm(np.array(A) - np.array(B))
    CA = np.linalg.norm(np.array(C) - np.array(A))
    R = BC * CA * AB / (4 * S)
    return R

def radius_predict(img,depth_image,d):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 对灰度图像进行二值化处理
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # # 显示二值化后的图像
    # cv2.imshow('Binary Image', thresh)
    # cv2.waitKey(0)
    # """
    # 具体来说，cv2.waitKey()函数的参数表示等待的毫秒数。
    # 如果指定为0，表示一直等待，直到用户按下任意键。
    # 如果指定为非零的值，表示等待指定的毫秒数后自动关闭窗口。"""
    # cv2.destroyAllWindows()

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # 开运算操作
    # """
    # 开运算是先腐蚀后膨胀的组合操作，可以去除二值图像中的小连通区域和突出部分。
    # 它通常用于去除噪声，平滑较小物体的轮廓以及断开物体之间的细小连接。
    # """
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # # 显示处理后的图像
    # cv2.imshow('Opening', opening)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 计算连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8, ltype=cv2.CV_16U)

    # 删除面积小于min_area的连通域
    min_area = 2000  # ......................................................可以根据经验更改
    for i in range(1, num_labels):  # 背景也是一个连通域
        area = stats[i, 4]  # 连通域面积
        # print(area)
        if area < min_area:
            opening[labels == i] = 0

    # # 显示处理后的图像
    # cv2.imshow('area_change', opening)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 获取骨架
    img = cv2.ximgproc.thinning(opening, thinningType=0)

    # # 反相图像（黑白颠倒显示）
    # img_copy1 = img.copy()
    # inverted_image = cv2.bitwise_not(img_copy1)
    # # 显示处理后的图像
    # cv2.imshow('Skeleton', inverted_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # 保存图像
    # cv2.imwrite('gu_img.jpg', inverted_image)


    # 将图像最外围像素值置零
    img[0, :] = 0  # 置零第一行
    img[-1, :] = 0  # 置零最后一行
    img[:, 0] = 0  # 置零第一列
    img[:, -1] = 0  # 置零最后一列
    h, w = img.shape[:2]
    img = img / 255

    # 获取线缆的交点和端点
    list_jiao = []
    list_duan = []


    for i in range(1, h - 1):
        for j in range(1, w - 1):
            """my_list[:-1]表示从列表的第一个元素开始，一直取到倒数第二个元素（不包括最后一个元素）"""
            p2, p3, p4, p5, p6, p7, p8, p9, p20 = neighbours = \
                [img[i - 1, j], img[i - 1, j + 1], img[i, j + 1], img[i + 1, j + 1],
                 img[i + 1, j], img[i + 1, j - 1], img[i, j - 1], img[i - 1, j - 1], img[i - 1, j]]
            if img[i, j] == 1 and 3 <= sum(neighbours[:-1]) and transitions(neighbours) >= 6:
                list_jiao.append((i, j))

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            p2, p3, p4, p5, p6, p7, p8, p9, p20 = neighbours = \
                [img[i - 1, j], img[i - 1, j + 1], img[i, j + 1], img[i + 1, j + 1],
                 img[i + 1, j], img[i + 1, j - 1], img[i, j - 1], img[i - 1, j - 1], img[i - 1, j]]
            if img[i, j] == 1 and sum(neighbours[:-1]) <= 2 and transitions(neighbours) <= 2:
                list_duan.append((i, j))
    # print(list_jiao)
    # print(list_duan)

    # img_draw = img.copy()
    # img_draw = 1 - img_draw
    # # 在图像中显示端点和交叉点(原来的)
    # # 指定圆心坐标和半径
    # for i in list_duan:
    #     center = (i[1],i[0])#x/列,y、行
    #     radius = 2
    #      # 指定颜色和线条宽度
    #     color = (0, 0, 0)
    #     thickness = 2
    #     # 在图像中绘制圆
    #     cv2.circle(img_draw, center, radius, color, thickness)
    #
    # for i in list_jiao:
    #     center = (i[1],i[0])
    #     radius = 2
    #      # 指定颜色和线条宽度
    #     color = (0, 0, 0)
    #     thickness = 2
    #     # 在图像中绘制圆
    #     cv2.circle(img_draw, center, radius, color, thickness)
    #
    # # img_draw = cv2.bitwise_not(img_draw)
    # # 显示图像
    # cv2.imshow('image', img_draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img_draw = (img_draw * 255).astype(np.uint8)
    # # 保存图像
    # cv2.imwrite('duan_img.jpg', img_draw)

    # 处理毛细分支
    mao_jiao = []
    zhen_jiao = []
    for i in range(len(list_jiao)):
        distanse = []
        for j in range(len(list_duan)):
            dis = math.sqrt((list_jiao[i][0] - list_duan[j][0]) ** 2 + (list_jiao[i][1] - list_duan[j][1]) ** 2)
            distanse.append(dis)
        if min(distanse) <= 40:
            mao_jiao.append(list_jiao[i])
        else:
            zhen_jiao.append(list_jiao[i])

    # print(zhen_jiao,mao_jiao)

    # 删除以交点为中心80像素长的正方形区域.............................................40根据实际进行修改
    for dian in zhen_jiao:
        for i in range(dian[0] - 40, dian[0] + 40):
            for j in range(dian[1] - 40, dian[1] + 40):
                if (h - 1) > i >= 1 and (w - 1) > j >= 1:
                    img[i, j] = 0

    # 删除以交点为中心5像素长的正方形区域.............................................40根据实际进行修改
    for dian in mao_jiao:
        for i in range(dian[0] - 5, dian[0] + 5):
            for j in range(dian[1] - 5, dian[1] + 5):
                if (h - 1) > i >= 1 and (w - 1) > j >= 1:
                    img[i, j] = 0


    # # 显示图像
    # inverted_image = 1 - img
    # cv2.imshow('image_xiu', inverted_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # 保存图像
    # inverted_image = (inverted_image * 255).astype(np.uint8)
    # cv2.imwrite('image_xiu.jpg', inverted_image)


    img_show = img.copy()
    # 去除交点附近区域后再寻找端点
    list_duan = []  # 行、列
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            p2, p3, p4, p5, p6, p7, p8, p9, p20 = neighbours = \
                [img[i - 1, j], img[i - 1, j + 1], img[i, j + 1], img[i + 1, j + 1],
                 img[i + 1, j], img[i + 1, j - 1], img[i, j - 1], img[i - 1, j - 1], img[i - 1, j]]
            if img[i, j] == 1 and sum(neighbours[:-1]) <= 2 and transitions(neighbours) <= 2:
                list_duan.append((i, j))
    # print(list_duan)

    """该函数对于浮点型数据类型的输入图像矩阵，
    先进行线性变换将像素值缩放到 [0, 255] 范围内，
    然后将像素值四舍五入取整并截断到 8 位无符号整数范围内"""
    img = cv2.convertScaleAbs(img)

    # 获取骨架分支编号和像素坐标
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

    # def generate_random_color():
    #     """生成随机颜色"""
    #     return np.random.randint(0,256,3)
    #
    # def fill_color(img1,n,img2):
    #     """
    #     为不同的连通域填色
    #     """
    #     h,w = img1.shape
    #     res = np.zeros((h,w,3),img1.dtype)
    #     #生成随机颜色
    #     random_color = {}
    #     for c in range(1,n):
    #         random_color[c] = generate_random_color()
    #     #为不同的连通域填色
    #     for i in range(h):
    #         for j in range(w):
    #             item = img2[i][j]
    #             if item == 0:
    #                 pass
    #             else:
    #                 res[i,j,:] = random_color[item]
    #     return res
    #
    # #为不同连通域着色
    # img_label = fill_color(img,num_labels,labels)
    # # 显示图像
    # cv2.imshow('image'
    #            '_labels', img_label)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 获取骨架分支的像素坐标
    branch_coords = []
    for i in range(1, num_labels):
        mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(contours)#nx1x2
        coords_nmlist = np.concatenate(contours, axis=0)[:, 0, :]
        # print(coords)
        # 去除毛细分支
        if len(coords_nmlist) >= 80:
            branch_coords.append(coords_nmlist)

    # for i, coord in enumerate(branch_coords):
    #     print(f"Branch {i}: {coord}")

    for i, coords in enumerate(branch_coords):
        if (coords[0][1], coords[0][0]) not in list_duan and (coords[-1][1], coords[-1][0]) not in list_duan:
            # print(coords)
            found = False
            for duan in list_duan:
                if contains_point(coords, duan[1], duan[0]) and not found:
                    start_point = duan
                    found = True
            # print(start_point)
            start_index = np.where((coords[:, 0] == start_point[1]) & (coords[:, 1] == start_point[0]))[0].item()
            # print(start_index)
            coords = np.concatenate((coords[start_index:], coords[:start_index]), axis=0)

        branch_coords[i] = coords[:len(coords) // 2]
        # print(f"Branch {i}: {branch_coords[i]}")

    # dis = depth_frame.get_distance(ux, uy)#(ux,uy)分别为宽高，所得深度单位是m。
    # """rs.rs2_deproject_pixel_to_point需要深度相机的内参来进行像素点到相机坐标系的转换。"""
    # camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)  # 计算相机坐标系xyz
    # camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
    # camera_xyz = camera_xyz.tolist()

    # 从文件中读取深度图像内参矩阵字典
    with open('../depth_intrin.pickle', 'rb') as f:
        depth_intrin_dict = pickle.load(f)

    # 将字典对象转换为rs2_intrinsics对象
    depth_intrin = rs.intrinsics()
    depth_intrin.width = depth_intrin_dict['width']
    depth_intrin.height = depth_intrin_dict['height']
    depth_intrin.ppx = depth_intrin_dict['ppx']
    depth_intrin.ppy = depth_intrin_dict['ppy']
    depth_intrin.fx = depth_intrin_dict['fx']
    depth_intrin.fy = depth_intrin_dict['fy']
    depth_intrin.model = depth_intrin_dict['model']
    depth_intrin.coeffs = depth_intrin_dict['coeffs']

    radius = {}
    radius_detail = {}

    for i, branch in enumerate(branch_coords):
        radius_list = []
        mid_point = (branch[len(branch) // 2][0],branch[len(branch) // 2][1])  # (x,y)
        l = len(branch) // d  # ...............................................可以根据实际值进行修改
        # print(l)
        if l >= 3:
            for j in range(1, l - 1):
                dian_1 = branch[d * (j - 1)]
                dian_2 = branch[d * j]
                dian_3 = branch[d * (j + 1)]
                depth1 = depth_image[dian_1[1], dian_1[0]]
                depth2 = depth_image[dian_2[1], dian_2[0]]
                depth3 = depth_image[dian_3[1], dian_3[0]]
                # print(dian_1, dian_2, dian_3, depth1, depth2, depth3)
                if depth1 and depth2 and depth3:  # 高度值为空时半径为None
                    dian_1_3d = rs2_deproject_pixel_to_point(depth_intrin, dian_1, depth1)
                    dian_2_3d = rs2_deproject_pixel_to_point(depth_intrin, dian_2, depth2)
                    dian_3_3d = rs2_deproject_pixel_to_point(depth_intrin, dian_3, depth3)
                    r = circle_radius(dian_1_3d, dian_2_3d, dian_3_3d)
                    # print(dian_1_3d,dian_2_3d,dian_3_3d,r)
                else:
                    r = None
                if r:
                    radius_list.append(r)
            dian_1 = branch[d * (l-2)]
            dian_2 = branch[d * (l-1)]
            dian_3 = branch[d * l-1]
            depth1 = depth_image[dian_1[1], dian_1[0]]
            depth2 = depth_image[dian_2[1], dian_2[0]]
            depth3 = depth_image[dian_3[1], dian_3[0]]
            # print(dian_1, dian_2, dian_3, depth1, depth2, depth3)
            if depth1 and depth2 and depth3:  # 高度值为空时半径为None
                dian_1_3d = rs2_deproject_pixel_to_point(depth_intrin, dian_1, depth1)
                dian_2_3d = rs2_deproject_pixel_to_point(depth_intrin, dian_2, depth2)
                dian_3_3d = rs2_deproject_pixel_to_point(depth_intrin, dian_3, depth3)
                r = circle_radius(dian_1_3d, dian_2_3d, dian_3_3d)
                # print(dian_1_3d,dian_2_3d,dian_3_3d,r)
            else:
                r = None
            if r:
                radius_list.append(r)
        elif 2*d < len(branch) < 3*d:
            dian_1 = branch[0]
            dian_2 = branch[d]
            dian_3 = branch[2*d]
            depth1 = depth_image[dian_1[1], dian_1[0]]
            depth2 = depth_image[dian_2[1], dian_2[0]]
            depth3 = depth_image[dian_3[1], dian_3[0]]
            # print(dian_1, dian_2, dian_3, depth1, depth2, depth3)
            if depth1 and depth2 and depth3:  # 高度值为空时半径为None
                dian_1_3d = rs2_deproject_pixel_to_point(depth_intrin, dian_1, depth1)
                dian_2_3d = rs2_deproject_pixel_to_point(depth_intrin, dian_2, depth2)
                dian_3_3d = rs2_deproject_pixel_to_point(depth_intrin, dian_3, depth3)
                r = circle_radius(dian_1_3d, dian_2_3d, dian_3_3d)
                # print(dian_1_3d, dian_2_3d, dian_3_3d, r)
            else:
                r = None
            if r:
                radius_list.append(r)

        # if len(radius_list) >= 1:
        #     #显示折弯半径集图像
        #     for k in range(len(radius_list)):
        #         if radius_list[k] >= 1000 or radius_list[k] == 0.0:
        #             radius_list[k] = radius_list[k-1]
        #     x40 = np.arange(40, (len(radius_list) + 1) * 40, 40)
        #     plt.plot(x40, radius_list, label='Line')
        #     plt.xlabel('points')
        #     plt.ylabel('radius/mm')
        #     plt.title('{}'.format(branch[0]))
        #     plt.legend()
        #     plt.show()
        if radius_list:
            # 可以加上对半径值计算结果过滤后的结果
            radius[mid_point] = "{:.1f}".format(min(radius_list))
            radius_detail[mid_point] = radius_list
        else:
            radius[mid_point] = None
            radius_detail[mid_point] = None
    return radius,img_show,radius_detail,d

if __name__ == "__main__":

    #..................................................进行预测时，注意图像信息和深度信息要对应起来

    deeplab = DeeplabV3()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    # -------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    count = False
    name_classes = ["background", "line"]

    # '''
    #     predict.py有几个注意点
    #     1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
    #     具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
    #     2、如果想要保存，利用r_image.save("img.jpg")即可保存。
    #     3、如果想要原图和分割图不混合，可以修改deeplab的mix_type。
    #     4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
    #     seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
    #     for c in range(self.num_classes):
    #         seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
    #         seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
    #         seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
    # '''
    while True:
        img = input('Input image filename:')
        depth_path = input('Input image filename:')
        d = input('distance:')

        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
            r_image = np.array(r_image)
            start_time = time.time()
            r_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Result', r_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # 从文本文件中加载深度图像数据
            depth_image = np.loadtxt(depth_path, delimiter=',')
            d = int(d)
            radius,img,radius_detail,d = radius_predict(r_image,depth_image,d)
            print(radius)
            for key, value in radius_detail.items():
                print(key, value)
            for mid_point,radiu in radius.items():
                # 将文本渲染到图像上
                cv2.putText(img, str(radiu), mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            # # 指定圆的中心坐标和半径
            # center = (d, d)  # 左上角
            # radius = d
            # # 指定颜色（BGR 格式）
            # color = (255, 255, 255)  # 红色
            #
            # # 在图像上画圆
            # cv2.circle(img, center, radius, color, thickness=2)  # thickness=-1 表示填充圆

            end_time = time.time()
            total_time = end_time - start_time
            print("程序运行时间为：{:.2f}秒".format(total_time))
            cv2.imshow('Result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()