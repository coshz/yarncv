import alibabacloud_oss_v2 as oss 
import os 
from concurrent.futures import ThreadPoolExecutor

__all_ = [ 'ImageUploader']


OSS_REGION = 'cn-hangzhou'
OSS_ENDPOINT = 'oss-cn-hangzhou.aliyuncs.com'
OSS_BUCKET = 'xzq-yarn'

def make_osskey(img_path):
    return os.path.basename(img_path)

def make_client_and_bucket(region, endpoint, bucket): 
    cfg = oss.config.load_default() 
    # assume key and secret is set in environment
    cfg.credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider() 
    cfg.region = region
    cfg.endpoint = endpoint
    return oss.Client(cfg), bucket


def put_image_into_oss(client, bucket_name, key, image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
        result = client.put_object(oss.PutObjectRequest(
            bucket=bucket_name,
            key=key,
            body=data
        ))
        return result
    if result.status_code != 200:
        raise f"put error: {result.status_code}"


def get_image_from_oss(client, bucket_name, key, download_path):
    result = client.get_object(oss.GetObjectRequest(
        bucket=bucket_name,
        key=key
    ))
    if result.status_code == 200:
        with result.body as stream:
            with open(download_path, 'wb') as f:
                f.write(stream.read())
    else:
        raise f"get error: {result.status_code}"


class ImageUploader:
    def __init__(self, max_workers=4):
        self.client, self.bucket = make_client_and_bucket(OSS_REGION, OSS_ENDPOINT, OSS_BUCKET) 
        self.executor = ThreadPoolExecutor(max_workers)

    def upload_async(self, img_path):
        self.executor.submit(self.upload, img_path)

    def upload(self, img_path):
        put_image_into_oss(self.client, self.bucket, make_osskey(img_path), img_path)