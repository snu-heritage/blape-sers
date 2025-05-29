import hashlib, requests, zipfile, sys
from pathlib import Path

# ------------- Zenodo 설정 ------------- #
RECORD_ID = "15487399"
BASE_URL  = f"https://zenodo.org/record/{RECORD_ID}/files"

FILES = {
    "raw.zip": {
        "flag": "raw",
        "md5": "f0e6341ec9ba3519127648324758904a",
    },
    "baseline_removed.zip": {
        "flag": "baseline_removed",
        "md5": "f6f7d5fa40f26b3cd1c6ca7826cca7e5",
    },
}


# ------------ 유틸리티 ------------ #
def _md5(fname, chunk=1 << 20):
    import hashlib, io
    h = hashlib.md5()
    with open(fname, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

def _bar(cur, total, w=40):
    done = int(w * cur / total)
    sys.stdout.write(f"\r[{'#'*done}{'-'*(w-done)}] {cur/total:6.2%}")
    sys.stdout.flush()

def zenodo_download(fname: str, dest: Path):
    url = f"{BASE_URL}/{fname}?download=1"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    _bar(downloaded, total)
    _bar(total, total); print()   # 끝줄

# ------------ 메인 ------------ #
def download_data(path='data', raw=True, baseline_removed=True):
    targets = [
        name for name, meta in FILES.items()
        if (raw and meta["flag"] == "raw")
        or (baseline_removed and meta["flag"] == "baseline_removed")
    ]
    if not targets:
        print("No data selected."); return None

    data_dir = Path(path); data_dir.mkdir(exist_ok=True)

    for name in targets:
        zip_path = data_dir / name
        extract_dir = data_dir / Path(name).stem   # zip 이름 == 폴더 이름
        if extract_dir.exists():
            print(f"[=] {extract_dir} already exists → skip")
            continue

        print(f"[+] Downloading {name}")
        zenodo_download(name, zip_path)

        # 무결성
        if _md5(zip_path) != FILES[name]["md5"]:
            raise RuntimeError(f"Checksum mismatch for {name}")

        # 압축 해제
        print(f"    Extracting to {extract_dir}/ …")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
        zip_path.unlink()
        print("    Done. ZIP removed.")

    print("All requested data ready.")
    return str(data_dir.resolve())


if __name__ == "__main__":
    download_data() 