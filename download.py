import gdown

from utils import *

settings = get_settings()
globals().update(settings)

des = path_join(route, 'download')
mkdir(des)

url = "https://drive.google.com/file/d/1m5DYZzoL9moPFzTZDr501tMr_n4U48HU/view?usp=share_link"
output = f"{des}/facepose_dataset.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)