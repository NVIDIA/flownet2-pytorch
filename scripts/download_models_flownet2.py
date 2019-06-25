from pathlib import Path
from download_gdrive import download_file_from_google_drive

file_id = '1E8re-b6csNuo-abg1vJKCDjCzlIam50F'
chpt_path = Path.cwd() / 'checkpoints'
destination = str(chpt_path / 'FlowNet2_checkpoint.pth.tar')
download_file_from_google_drive(file_id, destination)
