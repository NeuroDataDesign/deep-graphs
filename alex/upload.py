from tqdm import tqdm
from glob import glob
import argparse
import numpy as np
from cloudvolume import CloudVolume
from pathlib import Path
import tifffile as tf
from joblib import Parallel, delayed, cpu_count


def create_image_layer(destination, num_resolutions):
	info = CloudVolume.create_new_info(
		num_channels 	= 1,
		layer_type	= 'image',
		data_type	= 'uint16',
		encoding	= 'raw',
		resolution	= [1,1,1],
		voxel_offset	= [0, 0, 0],
		chunk_size	= [52, 100, 132],
		volume_size	= [208, 400, 528]
	)

	vol = CloudVolume(destination, info=info, parallel=True, compress=False)
	print(vol.info)
	vol.commit_info()
	return vol

#python upload.py **source** **destination**
def main():
	parser = argparse.ArgumentParser('Convert a folder of tif files to neuroglancer format')
	parser.add_argument('source', help='source path')
	parser.add_argument('destination', help='destination path')
	num_resolutions = 1

	args = parser.parse_args()

	files = [str(i) for i in Path(args.source).rglob('*.tif')]
	vol = create_image_layer(args.destination, num_resolutions)

	pbar = tqdm(enumerate(files), total=len(files))
	for idx,item in pbar:
		pbar.set_description_str('uploading chunks to resolution {}...'.format(num_resolutions - idx - 1))
		tf.imread(item)
		img = np.squeeze(tf.imread(item))
		#print(img[,:,:,:].shape)
		#img = np.expand_dims(img[0,:,:,:],2)
		img = np.expand_dims(img[:,:,:],3)
		print(vol[:,:,:].shape)
		print(img.shape)
		vol[:,:,:] = img

if __name__ == "__main__":
	main()
