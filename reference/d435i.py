import pyrealsense2 as rs
import numpy as np
import cv2
import pickle

# 创建Realsense管道
pipeline = rs.pipeline()

#创建配置对象
config = rs.config()
# # # 定义孔填充过滤器
# hole_filling = rs.hole_filling_filter()
#启用RGB和深度流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# 开始管道
profile = pipeline.start(config)


# 创建对齐对象
"""
将RGB图像和深度图像对齐是将这些不同的信息来源与其在3D空间中的位置进行精确匹配的过程。
这可以使计算机视觉应用程序（例如对象识别、3D建模和增强现实）更加准确和有效，
因为它们可以利用RGB和深度数据来更好地了解场景的结构和属性。
"""
align = rs.align(rs.stream.color)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)#Depth Scale is:  0.0010000000474974513

try:
    # 循环读取帧
    while True:
        # 等待一个新的帧
        frames = pipeline.wait_for_frames()

        # 对齐深度图和RGB图
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        # dis = depth_frame.get_distance(200, 300) # (ux,uy)分别为宽高
        # print(dis)#单位是m,除以depth scale单位是mm
        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        print('color',intr)
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics # 获取深度参数（像素坐标系转相机坐标系会用到）
        print('depth',depth_intrin)
        # 将rs2_intrinsics对象转换为字典
        depth_intrin_dict = {
            'width': depth_intrin.width,
            'height': depth_intrin.height,
            'ppx': depth_intrin.ppx,
            'ppy': depth_intrin.ppy,
            'fx': depth_intrin.fx,
            'fy': depth_intrin.fy,
            'model': depth_intrin.model,
            'coeffs': depth_intrin.coeffs
        }

        # 保存深度图像内参矩阵到文件
        with open('../depth_intrin.pickle', 'wb') as f:
            pickle.dump(depth_intrin_dict, f)

        # depth_frame = hole_filling.process(depth_frame)


        # 获取RGB图像
        color_image = np.asanyarray(color_frame.get_data())
        # 获取对齐的深度图像
        depth_image = np.asanyarray(depth_frame.get_data())

        # 将深度图像保存到磁盘上
        np.savetxt("depth.txt", depth_image,delimiter=',')
        cv2.imwrite("color_image.png",color_image)
        # 显示RGB图像
        cv2.imshow("RGB", color_image)

        # 显示深度图像
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("Depth", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            """
            cv2.waitKey(1)是OpenCV中的一个函数，
            它会在指定的毫秒数后返回-1（如果没有键被按下）或按下的键的ASCII码。在此代码中，它每1毫秒调用一次。
            
            在Python中，0xFF是一个十六进制数，它表示二进制数11111111，即8个二进制位都设置为1。
            在OpenCV中，它通常用于提取按下的键的最低8位ASCII码，通过按位与运算符（&）将提取的结果与0xFF进行按位与操作，可以获得ASCII码的最低8位，而忽略任何高位。这是因为ASCII码范围为0-127，可以用7个二进制位来表示，
            因此最高位总是0。通过按位与0xFF，我们确保任何高于8位的值都被设置为0，只提取最低8位的值。
            
            ord('q')将字符“q”转换为对应的ASCII码。
            因此，当用户按下“q”键时，该代码将返回一个非零值，此时程序将跳出循环并终止。
            因此，无论是 ASCII 编码还是 Unicode 编码，都可以使用 ord() 函数将字符 'q' 转换为相应的码点值。
            """
            break

finally:
    # 关闭管道
    pipeline.stop()

# 关闭所有窗口
cv2.destroyAllWindows()
