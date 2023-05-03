import sys

if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1:
        folders_path = args[1]
        bboxes_paths = args[1]
    else:
        raise AssertionError("Expected the folder with sequence as an input")


