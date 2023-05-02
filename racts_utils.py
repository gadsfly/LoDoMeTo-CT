import h5py
import numpy as np
from pathlib import Path
import os
from skimage.transform import iradon, radon

# Functions for image processing and placing dobjs

def normalize_img(img):
    # Find the minimum and maximum pixel values in the img
    min_pixel_value = np.min(img)
    max_pixel_value = np.max(img)

    # Normalize the img to the range [0, 1]
    normalized_img = (img - min_pixel_value) / (max_pixel_value - min_pixel_value)

    return normalized_img


def random_walk(num_steps, radius):
    walk = [(0, 0)]

    for _ in range(num_steps):
        dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        new_x, new_y = walk[-1][0] + dx, walk[-1][1] + dy

        if np.sqrt(new_x**2 + new_y**2) <= radius:
            walk.append((new_x, new_y))

    return walk

def create_dobj(radius, irregular=False):
    x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
    distance = x**2 + y**2
    dobj = np.where(distance <= radius**2, 1, 0)

    if irregular:
        walk = random_walk(3 * radius, radius)
        irregular_dobj = np.zeros((2 * radius + 1, 2 * radius + 1))

        for point in walk:
            irregular_dobj[point[1] + radius, point[0] + radius] = 1

        dobj = irregular_dobj

    return dobj


def insert_dobj(x_ray, dobj, x=None, y=None, random_location=False, intensity=1):
    x_ray = x_ray.copy()
    dobj = dobj.astype(float)
    dobj_height, dobj_width = dobj.shape
    x_ray_height, x_ray_width = x_ray.shape

    if random_location:
        x = np.random.randint(dobj_width // 2, x_ray_width - dobj_width // 2 - 1)
        y = np.random.randint(dobj_height // 2, x_ray_height - dobj_height // 2 - 1)

    x_start, x_end = max(0, x - dobj_width // 2), min(x_ray_width, x + dobj_width // 2 + 1)
    y_start, y_end = max(0, y - dobj_height // 2), min(x_ray_height, y + dobj_height // 2 + 1)

    dobj_x_start, dobj_x_end = max(0, dobj_width // 2 - x), min(dobj_width, dobj_width // 2 + x_ray_width - x)
    dobj_y_start, dobj_y_end = max(0, dobj_height // 2 - y), min(dobj_height, dobj_height // 2 + x_ray_height - y)

    # create mask so the 
    dobj_mask = np.zeros(x_ray.shape)
    dobj_mask[y_start:y_end, x_start:x_end] = dobj[dobj_y_start:dobj_y_end, dobj_x_start:dobj_x_end]
    dobj_mask = dobj_mask.astype(bool)
    
    x_ray[dobj_mask] = 0
#     x_ray[y_start:y_end, x_start:x_end] = 0
    dobj[dobj_y_start:dobj_y_end, dobj_x_start:dobj_x_end] *= intensity
    x_ray[y_start:y_end, x_start:x_end] += dobj[dobj_y_start:dobj_y_end, dobj_x_start:dobj_x_end]
    
    return x_ray

def sample_sino(sinogram, sample_rate):
    
    # get n samplings
    n_s = sinogram.shape[1]
    
    # get step size
    step_size = 1/sample_rate

    # get indices not being sampled
    indices = np.arange(0, n_s, step_size).astype(int)
    theta_inds = np.arange(0, n_s)
    zero_inds = list(set(theta_inds) - set(indices)) # indices to be set to 0
    
    # set non-sampled indices to 0
    sampled_sinogram = sinogram.copy()
    sampled_sinogram[:, zero_inds] = 0
    
    return sampled_sinogram, indices

### SAVE FILES

def raw2sino(raw_img, dobj=True, dobj_value=5.16, radius=20, spread=10, n_samples=362, circle=False):
    """
    dobj_value of 5.16 corrsponds to steel
    """

    # put dobj into the img
    poss_radii = np.arange(radius-spread,radius+spread+1)
    dobj_radius = np.random.choice(poss_radii, size=1)[0]
    dobj = create_dobj(dobj_radius, irregular=False)
    dobj_img = insert_dobj(raw_img, dobj, random_location=True, intensity=dobj_value)

    # get sinogram of dobj img
    theta = np.linspace(0., 180., n_samples, endpoint=False)
    dobj_sinogram = radon(dobj_img, theta=theta, circle=circle)
    
    return dobj_img, dobj_sinogram, radius

def get_path_dict():
    ct_dct = {}
    paths = list(Path("/work/bel25/datasets/").iterdir())
    paths = sorted(paths, key= lambda x : x.stem.split('_')[-1])

    for dset in ['train', 'val', 'test']:
        ct_dct[dset] = {}
        for mode in ['ground_truth', 'observation']:
            ct_dct[dset][mode] = list(filter(lambda x : dset in x.stem and mode in x.stem, paths ))

    return ct_dct

def save_dict_as_group(hdf5_group, dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            sub_group = hdf5_group.create_group(key)
            save_dict_as_group(sub_group, value)
        else:
            hdf5_group.attrs[key] = value

def save_files(dset, version=0, max_n=200, 
               dobj_value=5.16, 
               radius=20, 
               spread=10, 
               n_samples=512, #362, 
               circle=False,
               test_run=False):
    
    print('Beginning saving files for {} dataset'.format(dset))
    
    # Make directory
    output_dir = '/work/bel25/sim_datasets/{}_dobj_v{}'.format(dset, version)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # get paths
    paths = get_path_dict()[dset]['ground_truth']
    
    ct = 0
    total_ct = 0
    for p in paths:

        # load in file
        f = h5py.File(p, 'r')
        imgs = f['data']
        
        # skip img if possible
#         file_ind = int(p.stem.split('_')[-1])
#         inds = [file_ind + i  for i in range(imgs.shape[0])]
        file_ind = int(p.stem.split('_')[-1])
        inds = [total_ct + i  for i in range(imgs.shape[0])]
        img_paths = [os.path.join(output_dir, 'sample_{}.h5'.format(ind)) for ind in inds]
        if all(os.path.exists(p) for p in img_paths):
            total_ct += imgs.shape[0] # add to keep running ct to keep track of proper indexing
            print('first:', img_paths[0])
            print('last:', img_paths[-1])
            print("all paths found for", p.name)
            continue

        for i, img in enumerate(imgs):

            total_ct += 1
            hdf5_file_path = os.path.join(output_dir, 'sample_{}.h5'.format(total_ct))
#             ind = file_ind + i
#             hdf5_file_path = os.path.join(output_dir, 'sample_{}.h5'.format(ind))
            if os.path.exists(hdf5_file_path):
#                 print("img {},{} has been skipped".format(file_ind, i))
                continue

            if img.shape != (362,362):
                print("img {},{} is wrong shape of {}".format(file_ind, i, img.shape))
                continue
                
            # get artifact image and sinogram
            if not test_run:
                dobj_img, sino, sample_radius = raw2sino(img, dobj=True, 
                                          dobj_value=dobj_value, 
                                          radius=radius, 
                                          spread=spread, 
                                          n_samples=n_samples, 
                                          circle=circle)

                # Save the dictionary
                save_settings = {
                    "transform": {
                        "module": "sklearn",
                        "circle" : circle,
                        "n_samples" : n_samples
                    },
                    "dobj": {
                        "value" : dobj_value,
                        "radius" : sample_radius, 
                    }
                }

            # Create an hdf5 file
            if not test_run:
                with h5py.File(hdf5_file_path, 'w') as hdf5_file:
                    # Save the image
                    hdf5_file.create_dataset('ground_truth', data=dobj_img)
                    hdf5_file.create_dataset('sinogram', data=sino)
                    # Save the dictionary as a group with attributes
                    dict_group = hdf5_file.create_group('settings')
                    save_dict_as_group(dict_group, save_settings)        
                    print("Saved to: ", hdf5_file_path)
            else:
                print("TEST... Saved to: ", hdf5_file_path)
            
            # augment current count (number actually being saved)
            ct += 1
            
            if ct == max_n:
                break
        if ct == max_n:
            break
    print('Done: Saved {} files'.format(ct))
    print()

            
def load_dict_from_group(hdf5_group):
    dictionary = {}
    for key, value in hdf5_group.attrs.items():
        dictionary[key] = value
    for subgroup_name, subgroup in hdf5_group.items():
        if isinstance(subgroup, h5py.Group):
            dictionary[subgroup_name] = load_dict_from_group(subgroup)
    return dictionary


def load_sim_data(sample_index, dset='train', version=0):
    output_dir = '/work/bel25/sim_datasets/{}_dobj_v{}'.format(dset, version)
    hdf5_file_path = os.path.join(output_dir, 'sample_{}.h5'.format(sample_index))
    if not os.path.exists(hdf5_file_path):
        raise ValueError("No file found for index {}".format(sample_index))
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Load the image
        image = hdf5_file['ground_truth'][:]
        sino = hdf5_file['sinogram'][:]
        
        # Load the dictionary
        dict_group = hdf5_file['settings']
        settings = load_dict_from_group(dict_group)
        
    return image, sino, settings


    ### P