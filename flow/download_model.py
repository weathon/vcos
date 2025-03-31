print("Downloading From Google Drive")
file_id="16ge0XYXekBkMvjb6WYbJuykFE94IWVJm"

import gdown
gdown.download(
    f"https://drive.google.com/uc?id={file_id}",
    "flow_model.pth")
print("Download Complete")