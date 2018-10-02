import shutil, os, pathlib, sys, time, re

def human_sorted(iterable):
    ''' Sorts the given iterable in the way that is expected. '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(iterable, key = alphanum_key)

class ElapsedTimer(object):
    def __init__(self,msg='Elapsed'):
        self.start_time = time.time()
        self.msg = msg
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print(self.msg + ": %s " % self.elapsed(time.time() - self.start_time),
              flush=True)
        return (self.msg + ": %s " % self.elapsed(time.time() - self.start_time))

def help_option(msg):
    if '-h' in sys.argv or '--help' in sys.argv:
        print(msg)
        exit()

def file_paths(root_dir_path):
    ''' generate file_paths of directory_path ''' 
    it = os.walk(root_dir_path)
    for root,dirs,files in it:
        for path in map(lambda name:os.path.join(root,name),files):
            yield path

def replace_part_of(srcpath, old_part, new_part):
    '''
    change old_part of srcpath with new_part
    old/new_part must be path or str(not include path delimiters) 
    '''
    p = pathlib.Path(srcpath)
    parts = [part.replace(old_part,new_part) for part in p.parts]
    return pathlib.Path(*parts)

def leaf_dir(dir_path):
    ''' get "old_parent" from "dir_path" '''
    #NOTE: Don't use os.path.split('a/b/c/') -> ('a/b/c', None)
    return pathlib.Path(dir_path).parts[-1]

def make_dstpath(srcpath, old_parent, new_ancestor):
    '''
    discard part of path until old_parent,
    and put new_ancestor into path

    ex) make_dstpath('a/b/cd/dx','cd','123/456') -> '123/456/dx'
    '''
    p = pathlib.Path(srcpath)
    idx = p.parts.index(old_parent)
    return str(pathlib.Path(new_ancestor) \
                      .joinpath(*p.parts[idx+1:]))

def safe_copytree(srcpath, dstpath, ignores):
    ''' 
    if dst_path doesn't exists,
    it creates new directory and copy all of contents under src
    except *ignores.

    src must exists.
    dst must not exists.

    # ex) copy directory structure without files.
    safe_copytree(src_dir_path, dst_dir_path, ['*.jpg', '*.jpeg', '*.png']) 
    '''
    try:
        shutil.copytree(srcpath, dstpath, 
                        ignore=shutil.ignore_patterns(*ignores))
    except Exception as e:
        print(e)

def safe_mv(file_path, dst_dir):
    '''
    If file_path doesn't exists, nothing happens.
    '''
    try:
        shutil.move(file_path, dst_dir)
    except Exception as e:
        #pass
        print(e)

import cv2
import numpy as np

def slice1channel(gray_rgb_img, channel=0):
    ''' gray_rgb_img is r=g=b image. '''
    grayscale_img = gray_rgb_img[:,:,channel]
    return grayscale_img.reshape(gray_rgb_img.shape[:2] + (1,))

def bgr_float32(uint8img):
    c = 1 if len(uint8img.shape) == 2 else 3
    h,w = uint8img.shape[:2]
    uint8img = (uint8img / 255).astype(np.float32)
    return uint8img.reshape((h,w,c))

def load_imgs(img_dir, mode_flag=cv2.IMREAD_GRAYSCALE):
    return map(lambda path: bgr_float32(cv2.imread(path, mode_flag)),
               human_sorted(file_paths(img_dir)))

def assert_not_exists(path):
    assert not os.path.exists(path), path + ' is already exists.' 
def assert_exists(path):
    assert os.path.exists(path), path + ' is not exists.' 

def binarization(img, threshold=100):
    binarized = (img >= threshold).astype(np.uint8) * 255
    return binarized

import unittest
class Test_replace_part_of_path(unittest.TestCase):
    def test_replace_part(self):
        p = pathlib.Path('root','bbb','ccc','leaf')
        self.assertEqual(replace_part_of(p,'root','xxxx'),
                         pathlib.Path('xxxx','bbb','ccc','leaf'))           
        self.assertEqual(replace_part_of(p,'leaf','xxxx'),
                         pathlib.Path('root','bbb','ccc','xxxx'))           
        self.assertEqual(replace_part_of(p,'bbb','123'),
                         pathlib.Path('root','123','ccc','leaf'))           

    def test_old_part_is_not_in_srcpath(self):
        # nothing happens.
        p = pathlib.Path('root','bbb','ccc','leaf')
        self.assertEqual(p,replace_part_of(p,'xx','asd'))

if __name__ == '__main__':
    print(make_dstpath('./asd/fde/gcd/sdf.jpg','gcd','./aaa/dx/'))
    print(make_dstpath('a/b/cd/dx','cd','123/456'))
    unittest.main()
