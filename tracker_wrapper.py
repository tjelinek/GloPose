import sys

if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1:
        file_name = args[1]
    else:
        raise AssertionError("Expected the folder with sequence as an input")


