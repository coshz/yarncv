from datetime import datetime
import time


def make_timestamp():
    # now = datetime.now()
    msts = round(1000 * time.time())
    # ts = now.strftime("%y%m%d%H%M%S") + f"{now.microsecond // 1000 :03d}"
    return str(msts)


def make_image_name(width, height, frame_num, timestamp, ext):
    return f"Image_w{width}_h{height}_fn{frame_num}_{timestamp}.{ext}"


if __name__ == '__main__':
    print(make_timestamp())