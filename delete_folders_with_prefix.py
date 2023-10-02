import shutil

def delete_folders_with_prefix(directory, prefix):
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs.copy():
            if dir_name.startswith(prefix):
                dir_to_delete = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_to_delete)
                except:
                  pass