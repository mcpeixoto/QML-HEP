"""
Author: Miguel Caçador Peixoto
Description: 
    This script is used to download the raw data from zenodo.
"""


# Imports
from os.path import join, basename
import os
import requests
import bs4 as bs
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool

from qmlhep.config import raw_data_path as save_path
from qmlhep.utils.helper import check_integrity


def download_file(ball, force_check=False, debug=False):
    url, save_path, lookuptable = ball

    """
    Download a file from a url and save it
    """

    temp_path = save_path + ".tmp"
    filename = basename(url)

    # If file not in sha256 table, that's because it does not exist
    if filename not in lookuptable.keys():
        print(f"[!] {filename} hash not found in lookup table!")
        return

    # If already downloaded & verified, skip
    if os.path.exists(save_path):
        if not force_check:
            if debug:
                print(f"[!] {filename} already downloaded. Skipping.")
            return
        if check_integrity(save_path, lookuptable):
            if debug:
                print(f"[!] {filename} already downloaded and verified. Skipping.")
            return

    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Download the file
    r = requests.get(url, stream=True)

    # Progress bar stuff
    total_size_in_bytes = int(r.headers.get("content-length", 0))
    chunk_size = 1024 * 4  # 1 Kibibyte

    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, desc=f"Downloading {filename}", leave=False, position=1, dynamic_ncols=True
    )

    with open(temp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                progress_bar.update(chunk_size)
                f.write(chunk)
                f.flush()

    # Check integrity
    if check_integrity(temp_path, lookuptable):
        os.rename(temp_path, save_path)
        if debug:
            print(f"[+] {filename} downloaded and verified")

    else:
        print(f"[-] {filename} corrupted! *****Please check manually*****")


if __name__ == "__main__":
    # We want all the pythia sanitized files
    zenodo_url = "https://zenodo.org/record/5126747#.YyH60tLMJcA"

    # Get Hash Table for integrity check
    soup = bs.BeautifulSoup(requests.get(zenodo_url).text, "html.parser")

    lookuptable = {}
    for container in soup.find_all("td"):
        try:
            filename = container.a.text
            if "h5" not in filename:
                continue
            md5 = container.small.text
            lookuptable[filename] = md5.replace(" ", "").replace("md5:", "")
        except:
            pass

    # Urls to download
    to_download = [f"https://zenodo.org/record/5126747/files/{x}" for x in lookuptable.keys() if "pythia_sanitised_features" in x]
    start_time = datetime.now()

    # Download all files with multiprocessing
    work = [(url, join(save_path, basename(url)), lookuptable) for url in to_download]
    with Pool(5, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        r = list(tqdm(p.imap(download_file, work), total=len(to_download), position=0, leave=True, dynamic_ncols=True, desc="Downloading all files"))

    end_time = datetime.now()
    print(f"> Downloaded {len(work)} files. Total time elapsed: {end_time - start_time}")
