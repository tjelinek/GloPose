#!/usr/bin/env python

import errno
import os
import sys
import zipfile

if sys.version_info[0] == 2:
    import urllib2
else:
    import urllib.request

data_url = 'https://www.eth3d.net/data/slam'


# Outputs the given text and lets the user input a response (submitted by
# pressing the return key). Provided for compatibility with Python 2 and 3.
def GetUserInput(text):
    if sys.version_info[0] == 2:
        return raw_input(text)
    else:
        return input(text)


# Creates the given directory (hierarchy), which may already exist. Provided for
# compatibility with Python 2 and 3.
def MakeDirsExistOk(directory_path):
    try:
        os.makedirs(directory_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# Gets a (single-digit) number input from the user.
def GetNumberInput(min_number, max_number):
    number = 0

    while True:
        response = GetUserInput("> ")
        if not response.isdigit():
            print('Please enter a number.')
        elif int(response) < min_number or int(response) > max_number:
            print('Pleaser enter a number in the given range.')
        else:
            print('')
            number = int(response)
            break

    return number


# Gets a single dataset name from the user.
def GetDatasetInput():
    print('Please enter the dataset name (see https://www.eth3d.net/slam_datasets for a list):')
    return GetUserInput("> ")


# Reads the content at the given URL and returns it as text.
def MakeURLRequest(url):
    url_object = None
    if sys.version_info[0] == 2:
        url_object = urllib2.urlopen(url)
    else:
        url_object = urllib.request.urlopen(url)

    result = ''

    block_size = 8192
    while True:
        buffer = url_object.read(block_size)
        if not buffer:
            break

        result += buffer.decode("utf-8")

    return result


# Downloads the given URL to a file in the given directory. Returns the
# path to the downloaded file.
# In part adapted from: https://stackoverflow.com/questions/22676
def DownloadFile(url, dest_dir_path):
    file_name = url.split('/')[-1]
    dest_file_path = os.path.join(dest_dir_path, file_name)

    url_object = None
    if sys.version_info[0] == 2:
        url_object = urllib2.urlopen(url)
    else:
        url_object = urllib.request.urlopen(url)

    with open(dest_file_path, 'wb') as outfile:
        meta = url_object.info()
        file_size = 0
        if sys.version_info[0] == 2:
            file_size = int(meta.getheaders("Content-Length")[0])
        else:
            file_size = int(meta["Content-Length"])
        print("Downloading: %s (size [bytes]: %s)" % (url, file_size))

        file_size_downloaded = 0
        block_size = 8192
        while True:
            buffer = url_object.read(block_size)
            if not buffer:
                break

            file_size_downloaded += len(buffer)
            outfile.write(buffer)

            sys.stdout.write(
                "%d / %d  (%3f%%)\r" % (file_size_downloaded, file_size, file_size_downloaded * 100. / file_size))
            sys.stdout.flush()

    return dest_file_path


# Unzips the given zip file into the given directory.
def UnzipFile(file_path, unzip_dir_path):
    zip_ref = zipfile.ZipFile(open(file_path, 'rb'))
    zip_ref.extractall(unzip_dir_path)
    zip_ref.close()


# Downloads a zip file and directly unzips it.
def DownloadAndUnzipFile(url, unzip_dir_path):
    archive_path = DownloadFile(url, unzip_dir_path)
    UnzipFile(archive_path, unzip_dir_path)
    os.remove(archive_path)


# Performs a request to get the list of all training datasets.
def GetTrainingDatasetList():
    text_list = MakeURLRequest(data_url + '/dataset_list_training.txt')
    return text_list.split('\n')


# Performs a request to get the list of all test datasets.
def GetTestDatasetList():
    text_list = MakeURLRequest(data_url + '/dataset_list_test.txt')
    return text_list.split('\n')


if __name__ == '__main__':
    print('=== ETH3D SLAM dataset download script ===')
    print('')
    print(
        'This script will download the datasets into \'training\' and \'test\' (and possibly \'calibration\') folders in the current working directory:')
    print(os.getcwd())
    print('If you want to download them into another directory, please re-start the script from that directory.')

    training_datasets_path = os.path.join('.', 'training')
    test_datasets_path = os.path.join('.', 'test')

    print('')
    print('Please enter a number to choose an action:')
    print('1) Download all datasets')
    print('2) Download all training datasets')
    print('3) Download all test datasets')
    print('4) Download a specific dataset')
    print('')

    number = GetNumberInput(1, 4)

    training_datasets = GetTrainingDatasetList()
    test_datasets = GetTestDatasetList()

    dataset_list = []
    if number == 1 or number == 2:
        dataset_list.extend(training_datasets)
    if number == 1 or number == 3:
        dataset_list.extend(test_datasets)
    if number == 4:
        dataset_list.append(GetDatasetInput())

    print('Please enter a number to choose the data to download:')
    print('1) Monocular RGB video (1 x undistorted RGB) in PNG format')
    print('2) Stereo RGB video (2 x undistorted RGB) in PNG format')
    print('3) RGB-D video (1 x undistorted RGB, 1 x undistorted depth) in PNG format (TUM RGB-D compatible)')
    print('4) All videos (2 x undistorted RGB, 1 x undistorted depth) in PNG format (TUM RGB-D compatible)')
    print('5) Raw data (2 x distorted RGB with Bayer pattern, 2 x distorted infrared) in rosbag format')
    print('')

    number = GetNumberInput(1, 5)

    download_mono = (number != 5)
    download_stereo = (number == 2 or number == 4)
    download_rgbd = (number == 3 or number == 4)
    download_imu = False
    download_raw = (number == 5)
    download_raw_calibration = False

    if number != 5:
        print('Please enter a number to choose whether to download IMU data:')
        print('1) Yes, download IMU data')
        print('2) No, download visual data only')
        print('')
        download_imu = (GetNumberInput(1, 2) == 1)

    if number == 5:
        print(
            'Please enter a number to choose whether to download the raw calibration subsequences (for training data):')
        print('1) Yes, download the per-dataset calibration subsequences')
        print('2) No, download the actual dataset subsequences only')
        print('')
        download_raw_calibration = (GetNumberInput(1, 2) == 1)

    for dataset in dataset_list:
        is_training_dataset = (dataset in training_datasets)
        if not is_training_dataset and not (dataset in test_datasets):
            print('Dataset not found in training or test dataset list, skipping: ' + dataset)
            continue

        datasets_path = training_datasets_path if is_training_dataset else test_datasets_path
        MakeDirsExistOk(datasets_path)

        if download_mono:
            DownloadAndUnzipFile(
                data_url + '/datasets/' + dataset + '_mono.zip',
                datasets_path)

        if download_stereo:
            DownloadAndUnzipFile(
                data_url + '/datasets/' + dataset + '_stereo.zip',
                datasets_path)

        if download_rgbd:
            DownloadAndUnzipFile(
                data_url + '/datasets/' + dataset + '_rgbd.zip',
                datasets_path)

        if download_imu:
            DownloadAndUnzipFile(
                data_url + '/datasets/' + dataset + '_imu.zip',
                datasets_path)

        if download_raw:
            DownloadAndUnzipFile(
                data_url + '/datasets/' + dataset + '_raw.zip',
                datasets_path)

            if download_raw_calibration and is_training_dataset:
                calibration_dataset_path = os.path.join(datasets_path, dataset, 'calibration_dataset.txt')
                if not os.path.isfile(calibration_dataset_path):
                    print('Error: No calibration_dataset.txt found for raw dataset: ' + dataset)
                else:
                    calibration_path = os.path.join('.', 'calibration')
                    MakeDirsExistOk(calibration_path)

                    calibration_dataset_name = ''
                    with open(calibration_dataset_path, 'rb') as infile:
                        calibration_dataset_name = infile.read().decode('UTF-8').rstrip('\n')

                    DownloadAndUnzipFile(
                        data_url + '/calibration/' + calibration_dataset_name + '.zip',
                        calibration_path)

    print('Finished.')
