import os, gdown, tarfile

def extract_tar(tar_file, dst_path, delete_after=False):
    tar = tarfile.open(tar_file)
    tar.extractall(dst_path).close()
    
    if delete_after:
        os.remove(tar_file)