import cv2
import os

class ScannetDataset:
    def __init__(self):
        self.path = "Z:/TeamFolder/GS_SLAM/ScanNet/scans/scene0000_00/out/"

        # TODO:
        self.img_pair = []
        self.rgb_list = []
        self.gray_list = []
        self.d_list = []

        self.rgb_width = []
        self.rgb_height = []
        self.d_width = []
        self.d_height = []
        self.d_shift = []
        self.rgb_intrinsic = []
        self.d_intrinsic = []

    def read_info_file(self, filename):
        file = open(filename)
        lines = file.readlines()
        keys_dict = {
            "m_colorWidth": self.rgb_width,
            "m_colorHeight": self.rgb_height,
            "m_depthWidth": self.d_width,
            "m_depthHeight": self.d_height,
            "m_depthShift": self.d_shift,
            "m_calibrationColorIntrinsic": self.rgb_intrinsic,
            "m_calibrationDepthIntrinsic": self.d_intrinsic
        }

        for line in lines:
            key, value = line.strip().split(' = ')
            if key in keys_dict:
                if '.' in value:
                    value_list = map(float, value.split())
                else:
                    value_list = map(int, value.split())
                keys_dict[key] += value_list

    def get_img_list(self):
        for filename in os.listdir(self.path):
            if filename.endswith(".color.jpg"):
                os.path.join(self.path, filename)
                rgb_image = cv2.imread(os.path.join(self.path, filename))
            elif filename.endswith(".depth.pgm"):
                depth_image = cv2.imread(os.path.join(self.path, filename))

    def get_camera_intrinsic(self):
        self.read_info_file(f'{self.path}_info.txt')
        # depth camera intrinsic
        fx = self.d_intrinsic[0]
        fy = self.d_intrinsic[5]
        cx = self.d_intrinsic[2]
        cy = self.d_intrinsic[6]
        return [fx, fy, cx, cy]

    def InitializeDataset(self):
        os.makedirs(f'{self.path}pair/rgb', exist_ok=True)
        os.makedirs(f'{self.path}pair/gray', exist_ok=True)
        os.makedirs(f'{self.path}pair/depth', exist_ok=True)

        cntr = 0
        for filename in os.listdir(self.path):
            print(filename)
            if filename.endswith(".color.jpg"):
                rgb = cv2.imread(os.path.join(self.path, filename))
                rgb_resized = cv2.resize(rgb, (self.d_width, self.d_height))
                cv2.imwrite(f'{self.path}pair/rgb/{str(cntr).zfill(5)}.png', rgb)

            elif filename.endswith(".depth.pgm"):
                d_image = cv2.imread(os.path.join(self.path, filename))

    def ReturnData(self, index):
        file_name = f'{str(index).zfill(5)}.png'
        rgb = cv2.imread(f'{self.path}pair/rgb/{file_name}')
        gray = cv2.imread(f'{self.path}pair/gray/{file_name}', cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(f'{self.path}pair/depth/{file_name}', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        return rgb, gray, d