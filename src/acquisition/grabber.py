import sys
sys.path.append("./MvImport")

from ctypes import *
from MvImport import *
from check_device import check_and_obtain_device_list
from upload_to_oss import ImageUploader
import time
from utils import *


def prepare_camera():
    MvCamera.MV_CC_Initialize()
    deviceList = check_and_obtain_device_list()

    nConnectionNum = 0

    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(
        deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)
    ).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        raise Exception("create handle fail! ret[0x%x]" % ret)

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        raise Exception("open device fail! ret[0x%x]" % ret)

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if (
        stDeviceList.nTLayerType == MV_GIGE_DEVICE
        or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE
    ):
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        raise Exception("set trigger mode fail! ret[0x%x]" % ret)

    # # ch:设置曝光时间 | en:Set exposure time
    # ret = cam.MV_CC_SetFloatValue("ExposureTime", 2000)
    # if ret != 0:
    #     raise Exception("set exposure time fail! ret[0x%x]" % ret)
    return cam 

def destory_camera(cam:MvCamera):
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()


class Grabber:
    def __init__(self, log_name):
        self.cam = prepare_camera()
        self.logger = make_logger(log_name)
        self.uploader = ImageUploader()

    def start(self, fps=30.0, out_dir="img/"):
        # ch:开始取流 | en:Start grab image
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise Exception("start grabbing fail! ret[0x%x]" % ret)
        
        # Read the frame rate of the camera
        # frame_rate_value = MVCC_FLOATVALUE() 
        # ret = self.cam.MV_CC_GetFloatValue("FrameRate", frame_rate_value)
        # if ret != 0:
        #     raise "Failed to get frame rate! ret[0x%x]" % ret

        # fps = min(fps, frame_rate_value.fCurValue)
        self.logger.info("start to save image with rate: %.2f FPS" % fps)

        try: 
            while True:
                st = time.time()
                img_path = self.save_image(4,out_dir)
                # self.uploader.upload_async(img_path)
                et = time.time() - st
                time.sleep(max(0, 1/fps - et))
        except KeyboardInterrupt:
            self.logger.info("Stopping capture by user keyboard interrupt...")
        finally:
            self.cam.MV_CC_StopGrabbing()
            destory_camera(self.cam)

    def save_image(self, save_type, out_dir):
        def resolve_type_and_name(save_type, frame_info):
            img_tables = {
                1: ('jpeg', MV_Image_Jpeg),
                2: ('bmp', MV_Image_Bmp),
                3: ('tif', MV_Image_Tif),
                4: ('png', MV_Image_Png)
            }
            ext, img_type = img_tables.get(save_type, img_tables[4])
            img_name = make_image_name(
                frame_info.stFrameInfo.nWidth,
                frame_info.stFrameInfo.nHeight,
                frame_info.stFrameInfo.nFrameNum,
                make_timestamp(),
                ext
            )
            return img_type, img_name

        # ch:获取的帧信息 | en: frame from device
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 20000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            self.logger.info(
                "get one frame: Width[%d], Height[%d], nFrameNum[%d]"
                % (
                    stOutFrame.stFrameInfo.nWidth,
                    stOutFrame.stFrameInfo.nHeight,
                    stOutFrame.stFrameInfo.nFrameNum,
                )
            )

            # ch: 保存至本地 | en: save to local
            img_type, img_name = resolve_type_and_name(save_type, stOutFrame)
            img_path = os.path.join(out_dir, img_name)
            if int(img_type) == 0: # raw format
                raise NotImplementedError
            else:
                ret = save_non_raw_image(img_type, stOutFrame, self.cam, img_path)
            if ret != 0:
                self.cam.MV_CC_FreeImageBuffer(stOutFrame)
                raise Exception("save image fail! ret[0x%x]" % ret)
            else:
                self.logger.info("Image saved")

            self.cam.MV_CC_FreeImageBuffer(stOutFrame)
        else:
            raise Exception("no data[0x%x]" % ret)
        
        return img_path


def save_non_raw_image(nSaveImageType, stOutFrame, cam_instance:MvCamera, img_path):
    stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
    stSaveParam.enPixelType = (
        stOutFrame.stFrameInfo.enPixelType
    )  # ch:相机对应的像素格式 | en:Camera pixel type
    stSaveParam.nWidth = stOutFrame.stFrameInfo.nWidth  # ch:相机对应的宽 | en:Width
    stSaveParam.nHeight = stOutFrame.stFrameInfo.nHeight  # ch:相机对应的高 | en:Height
    stSaveParam.nDataLen = stOutFrame.stFrameInfo.nFrameLen
    stSaveParam.pData = stOutFrame.pBufAddr
    stSaveParam.enImageType = (
        nSaveImageType  # ch:需要保存的图像类型 | en:Image format to save
    )
    stSaveParam.pcImagePath = create_string_buffer(img_path.encode("ascii"))
    stSaveParam.iMethodValue = 1
    stSaveParam.nQuality = 80  # ch: JPG: (50,99], invalid in other format
    return cam_instance.MV_CC_SaveImageToFileEx(stSaveParam)