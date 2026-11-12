"""Plan/download the official station archive. No gridded archive is required."""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path
from eupp_weather.common import digest, save_json
from eupp_weather.preprocessing import COUNTRIES

URL = 'https://zenodo.org/api/records/7708362/files/EUPPBench-stations.zip/content'
SIZE = 18324361137
MD5 = 'e409457279b3494d18f2dfb41f3f449b'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='data/raw')
    p.add_argument('--download', action='store_true', help='Explicitly download the 18.3 GB public archive')
    p.add_argument('--profile', choices=['temperature48', 'full'], default='temperature48')
    a = p.parse_args(); root = Path(a.root).resolve()
    print({'url': URL, 'archive_bytes': SIZE, 'md5': MD5, 'profile': a.profile,
           'note': 'Keep roughly 40 GB free for ZIP + extraction; prepared arrays require additional space.'}, flush=True)
    if not a.download:
        print('Plan only. Add --download on the machine that will hold the data.'); return
    root.mkdir(parents=True, exist_ok=True)
    archive = root/'EUPPBench-stations.zip'; partial = archive.with_suffix('.zip.partial')
    if not archive.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > SIZE: raise ValueError('Partial archive is too large')
        if offset < SIZE:
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=90) as response:
                if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise ValueError('Server did not honor resume range; preserve partial and retry later')
                with partial.open('ab' if offset else 'wb') as out:
                    while block := response.read(8*1024**2):
                        out.write(block); offset += len(block)
                        if offset % (256*1024**2) < len(block): print(f'Downloaded {offset/SIZE:.1%}', flush=True)
        if partial.stat().st_size != SIZE or digest(partial, 'md5') != MD5:
            raise ValueError('Archive size/checksum mismatch; incomplete file retained')
        partial.replace(archive)
    elif archive.stat().st_size != SIZE or digest(archive, 'md5') != MD5:
        raise ValueError('Existing archive checksum mismatch')
    types = ['surface'] if a.profile == 'temperature48' else ['surface', 'surface_postprocessed', 'pressure_500', 'pressure_700', 'pressure_850']
    names = set()
    for kind in ('ensemble_forecasts', 'ensemble_reforecasts', 'forecasts_observations', 'reforecasts_observations'):
        for typ in types:
            if 'observations' in kind and typ.startswith('pressure'): continue
            for country in COUNTRIES: names.add(f'stations_{kind}_{typ}_{country}.zarr')
    with zipfile.ZipFile(archive) as z:
        selected = [r for r in z.infolist() if Path(r.filename).parts and Path(r.filename).parts[0] in names]
        for row in selected:
            parts = Path(row.filename).parts
            if Path(row.filename).is_absolute() or '..' in parts or (row.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe archive entry')
            out = root/row.filename
            if row.is_dir(): out.mkdir(parents=True, exist_ok=True); continue
            if out.exists() and out.stat().st_size == row.file_size: continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(row) as source, out.with_suffix(out.suffix+'.partial').open('wb') as dest:
                shutil.copyfileobj(source, dest)
            out.with_suffix(out.suffix+'.partial').replace(out)
    # The archive's complete checksum is verified before extracting any data.
    save_json(root/f'download_{a.profile}.json', {'url': URL, 'bytes': SIZE, 'md5': MD5,
        'archive_verified': True, 'stores': sorted(names), 'data_license': 'See https://zenodo.org/records/7708362'})
    print(f'Verified and extracted {len(names)} requested Zarr stores under {root}', flush=True)


if __name__ == '__main__': main()

