import getpass
import os
import paramiko
import paramiko.ssh_exception
import sys
from tqdm import tqdm

# Server and filepaths
HOSTNAME = 'lasid-ws1.sor.ufscar.br'
OTHER_FILE = 'others'
FILE_NAMES = ['interactions.csv', 'users.csv', 'items.csv', OTHER_FILE]
DATASET_FOLDER = '/storage/recsys/1-datasets'
OUTPUR_DIR = 'datasets'
os.makedirs(OUTPUR_DIR, exist_ok=True)

# Class to speed up the download
class FastTransport(paramiko.Transport):
    def __init__(self, sock):
        super(FastTransport, self).__init__(sock)
        self.window_size = 2147483647
        self.packetizer.REKEY_BYTES = pow(2, 40)
        self.packetizer.REKEY_PACKETS = pow(2, 40)


# Show progress of file being downloaded
def print_progress(transferred_bytes, total_bytes):
    global progress_bar
    divisor = 1 # MB
    progress_bar.total = total_bytes / divisor
    progress_bar.n = transferred_bytes / divisor
    progress_bar.update(0)


# Download a single dataset recursively
def download_dataset_files(ftp_client, dataset_name, selected_files):
    global progress_bar
    dataset_filepath = os.path.join(DATASET_FOLDER, dataset_name).replace('\\','/')
    dataset_output = os.path.join(OUTPUR_DIR, dataset_name)
    os.makedirs(dataset_output, exist_ok=True)
    for file in ftp_client.listdir(path=dataset_filepath):
        if file in selected_files or (file not in FILE_NAMES and OTHER_FILE in selected_files):
            progress_bar = tqdm(
                desc=file,
                unit="B",
                unit_scale=True
            )
            ftp_client.get(
                remotepath=os.path.join(dataset_filepath, file).replace('\\','/'), 
                localpath=os.path.join(dataset_output, file),
                callback=print_progress
            )
            progress_bar.close()


# Download multiple datasets
def download_multiple_datasets(ftp_client, selected_datasets, selected_files):
    for i, dataset_name in enumerate(selected_datasets, start=1):
        print('\n({:02}/{:02}) Downloading {}...'.format(i, len(selected_datasets), dataset_name))
        download_dataset_files(ftp_client, dataset_name, selected_files)


# Creates SSH client
connected = False
connection_tries = 0
username = input('SSH username: ')
while not connected:
    password = getpass.getpass("{}@{}'s password: ".format(username, HOSTNAME))
    try:
        ssh_conn = FastTransport((HOSTNAME, 22))
        ssh_conn.use_compression(compress=True)
        ssh_conn.connect(username=username, password=password)
        connected = True
    except paramiko.ssh_exception.AuthenticationException:
        connection_tries += 1
        if connection_tries < 3:
            print('Permission denied, please try again.')
        else:
            print('Permission denied (publickey,password).')
            sys.exit(0)

# Creates FTP client
ftp_client = paramiko.SFTPClient.from_transport(ssh_conn)

# Filter datasets to download
print('\nAvailable datasets for download:')
available_datasets = sorted(ftp_client.listdir(path=DATASET_FOLDER))
for i, dataset_name in enumerate(available_datasets, start=1):
    print('[{}] {}'.format(i, dataset_name))
print('Select the datasets with their numbers separated by space (or press Enter to download them all):')
datasets_op = input().strip()

# Filter files to download
print('\nAvailable files for download:')
available_files = FILE_NAMES
for i, file_name in enumerate(available_files, start=1):
    print('[{}] {}'.format(i, file_name))
print('Select the files with their numbers separated by space (or press Enter to download them all):')
files_op = input().strip()
selected_files = FILE_NAMES if files_op == '' else [FILE_NAMES[int(f)-1] for f in files_op.split(' ')]

progress_bar = None

# Download datasets
if datasets_op == '': # all
    download_multiple_datasets(ftp_client, available_datasets, selected_files)
elif ' ' in datasets_op: # subset
    selected_datasets = [available_datasets[int(i)-1] for i in datasets_op.split(' ')]
    download_multiple_datasets(ftp_client, selected_datasets, selected_files)
else: # only one
    print(f'\nDownloading {available_datasets[int(datasets_op)-1]}...')
    download_dataset_files(ftp_client, available_datasets[int(datasets_op)-1], selected_files)
ftp_client.close()
print('\nDownload completed!')