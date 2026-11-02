"""Download Inside Airbnb `listings.csv.gz` snapshots (CC BY 4.0).

Inside Airbnb publishes a quarterly snapshot per city. Free downloads cover roughly the last
12 months, so old dates disappear. We therefore record the exact URLs and SHA-256 hashes in
`data/raw/MANIFEST.json` and publish the *processed* dataset to the Hugging Face Hub; that
copy is what makes the results reproducible.

Attribution: data from Inside Airbnb (https://insideairbnb.com), licensed CC BY 4.0.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
from rich.console import Console

from title_optimizer.config import DataConfig

GET_THE_DATA_URL = "https://insideairbnb.com/get-the-data/"
BASE_URL = "https://data.insideairbnb.com"
console = Console()


def listings_url(country: str, region: str, city: str, date: str) -> str:
    return f"{BASE_URL}/{country}/{region}/{city}/{date}/data/listings.csv.gz"


def resolve_latest_date(country: str, region: str, city: str, page_html: str | None = None) -> str:
    """Find the newest snapshot date for a city by reading the 'Get the Data' page."""
    if page_html is None:
        page_html = httpx.get(GET_THE_DATA_URL, timeout=60, follow_redirects=True).text
    pattern = re.escape(f"{BASE_URL}/{country}/{region}/{city}/") + r"(\d{4}-\d{2}-\d{2})/data/"
    dates = sorted(set(re.findall(pattern, page_html)))
    if not dates:
        raise ValueError(
            f"No snapshot found for {country}/{region}/{city}. Check the exact slugs on "
            f"{GET_THE_DATA_URL} (hover a city's listings.csv.gz link)."
        )
    return dates[-1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg: DataConfig, force: bool = False) -> list[Path]:
    raw_dir = Path(cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    page_html: str | None = None
    manifest, paths = [], []

    for c in cfg.cities:
        date = c.get("date", "latest")
        if date == "latest":
            if page_html is None:
                page_html = httpx.get(GET_THE_DATA_URL, timeout=60, follow_redirects=True).text
            date = resolve_latest_date(c["country"], c["region"], c["city"], page_html)
        url = listings_url(c["country"], c["region"], c["city"], date)
        out = raw_dir / f"{c['city']}_{date}_listings.csv.gz"

        if out.exists() and not force:
            console.print(f"[dim]exists[/] {out}")
        else:
            console.print(f"[bold]GET[/] {url}")
            with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
                r.raise_for_status()
                with out.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)

        manifest.append({**c, "date": date, "url": url, "file": out.name, "sha256": _sha256(out)})
        paths.append(out)

    (raw_dir / "MANIFEST.json").write_text(
        json.dumps(
            {
                "source": "Inside Airbnb (https://insideairbnb.com), CC BY 4.0",
                "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "files": manifest,
            },
            indent=2,
        )
    )
    return paths
