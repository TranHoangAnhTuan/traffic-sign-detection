from urllib.request import urlretrieve
from PIL import Image
from matplotlib import pyplot as plt

# save the background 
urlretrieve(background_url, background_file)

plt.imshow(Image.open(background_file))