from PIL import Image, ImageChops
im = Image.open('results/x.png')

def trim(image):
	bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
	diff =ImageChops.difference(image, bg)
	bbox = diff.getbbox()
	if not bbox:
		return image
   	return image.crop(bbox)
trim(im).show()
