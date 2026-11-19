"""Download explicit Commons candidates with attribution/licence and SHA256.

Source membership is fixed before predictions. Downloads never enter training
automatically. Venue roles keep development and external challenges apart.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import time


def clean(value):
    return html.unescape(re.sub('<[^>]*>', '', value)).strip()


def main():
    import requests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=Path('configs/online_source_candidates.json'))
    parser.add_argument('--output', type=Path, default=Path('data/online_sg_malls'))
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'VisionAidAcademicDatasetCuration/1.0 (public Commons image research)'
    params = dict(action='query', format='json', prop='imageinfo', iiprop='url|extmetadata|sha1|size|mime',
                  iiurlwidth=1280, titles='|'.join(c['title'] for c in candidates))
    response = session.get('https://commons.wikimedia.org/w/api.php', params=params, timeout=45)
    response.raise_for_status()
    raw = response.json()
    (args.output / 'commons_metadata.json').write_text(json.dumps(raw, indent=2))
    pages = {p['title']: p for p in raw.get('query', {}).get('pages', {}).values()}
    records = []
    for candidate in candidates:
        record = dict(candidate)
        try:
            info = pages[candidate['title']]['imageinfo'][0]
            meta = info['extmetadata']
            licence = clean(meta.get('LicenseShortName', {}).get('value', ''))
            if not (licence.startswith('CC BY') or licence in ('CC0', 'Public domain')):
                raise ValueError('No supported explicit reuse licence; manual review required')
            video = info['mime'].startswith('video/') or candidate['title'].endswith('.ogv')
            url = info['url'] if video else info.get('thumburl', info['url'])
            suffix = '.ogv' if video else Path(candidate['title']).suffix.lower()
            path = args.output / (candidate['id'] + suffix)
            if not path.exists():
                download = session.get(url, timeout=60)
                download.raise_for_status()
                if len(download.content) > 30_000_000:
                    raise ValueError('Candidate exceeds 30 MB acquisition cap')
                path.write_bytes(download.content)
            record.update(status='downloaded_pending_annotation', file=path.name,
                source_page=info['descriptionurl'], download_url=url, original_url=info['url'],
                original_sha1=info['sha1'], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                author=clean(meta.get('Artist', {}).get('value', 'Unknown')),
                licence=licence, licence_url=meta.get('LicenseUrl', {}).get('value', ''),
                modification='Downloaded original video' if video else 'Commons resized preview, requested width 1280; no crop',
                attribution=clean(meta.get('Attribution', {}).get('value', '')),
                annotation_status='pending', duplicate_audit_status='pending', model_prediction_used_for_selection=False)
        except (KeyError, IndexError, ValueError, requests.RequestException) as exc:
            record.update(status='blocked', error=str(exc))
        records.append(record)
        (args.output / 'sources.json').write_text(json.dumps(records, indent=2))
        print(candidate['id'], record['status'], flush=True)
        time.sleep(.2)
    lines = ['# Online-source attribution', '',
             'Public online sources, not original team photography. Roles were fixed before predictions.',
             'Preserve each source licence and attribution when redistributing images or adaptations.', '']
    for r in records:
        if 'file' in r:
            lines.append(f"- {r['id']}: {r['title']} by {r['author']}. {r['licence']} ({r['licence_url']}). "
                         f"Source: {r['source_page']}. Modification: {r['modification']}.")
    (args.output / 'ATTRIBUTION.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
