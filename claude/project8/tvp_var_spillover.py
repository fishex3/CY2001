import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class TvpVarSpilloverResult:
    spillover_targets: pd.DataFrame
    tci_daily: pd.DataFrame
    tci_quarterly: pd.DataFrame
    directional_spillover: pd.DataFrame


class TvpVarSpilloverError(RuntimeError):
    pass


def _normalize_price_index_for_r(prices_raw: pd.DataFrame) -> pd.DataFrame:
    df = prices_raw.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.strftime("%Y-%m-%d")
    return df


def _as_sector_arg(sectors: Iterable[str]) -> str:
    return ",".join([s for s in sectors if s])


def get_tvp_var_spillover(
    prices_raw: pd.DataFrame,
    sectors: list[str],
    vol_window: int,
    forecast_h: int,
    kappa1: float,
    kappa2: float,
    *,
    rscript_exe: str | None = None,
) -> dict:
    """
    Run TVP-VAR spillover via the R implementation in `r_packages/tvp_var_spillover.R`.

    This is a thin wrapper around `Rscript` (recommended over rpy2 on Windows).
    Returns a dict compatible with the previous `models.tvp_var_spillover.get_tvp_var_spillover`.
    """
    if prices_raw is None or prices_raw.empty:
        raise TvpVarSpilloverError("prices_raw is empty.")
    if not sectors:
        raise TvpVarSpilloverError("sectors is empty.")

    base_dir = Path(__file__).resolve().parent
    r_script_path = base_dir / "r_packages" / "tvp_var_spillover.R"
    if not r_script_path.exists():
        raise TvpVarSpilloverError(f"Missing R script at: {r_script_path}")

    rscript = rscript_exe or os.environ.get("RSCRIPT") or "Rscript"
    if shutil.which(rscript) is None:
        raise TvpVarSpilloverError(
            "Could not find Rscript on PATH. Install R and ensure `Rscript` is available, "
            "or set the `RSCRIPT` environment variable to the full path."
        )

    df_for_r = _normalize_price_index_for_r(prices_raw)

    with tempfile.TemporaryDirectory(prefix="tvp_var_") as tmp:
        tmp_path = Path(tmp)
        input_csv = tmp_path / "prices.csv"
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_for_r.to_csv(input_csv, index_label="date")

        args = [
            rscript,
            str(r_script_path),
            "--input",
            str(input_csv),
            "--output",
            str(out_dir),
            "--sectors",
            _as_sector_arg(sectors),
            "--vol_window",
            str(int(vol_window)),
            "--forecast_h",
            str(int(forecast_h)),
            "--kappa1",
            str(float(kappa1)),
            "--kappa2",
            str(float(kappa2)),
        ]

        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            debug = {
                "returncode": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-4000:],
                "stderr_tail": (proc.stderr or "")[-4000:],
            }
            raise TvpVarSpilloverError(
                "R spillover computation failed.\n" + json.dumps(debug, indent=2)
            )

        def read_csv(name: str) -> pd.DataFrame:
            p = out_dir / name
            if not p.exists():
                raise TvpVarSpilloverError(f"Expected R output missing: {p}")
            return pd.read_csv(p)

        spillover_targets = read_csv("spillover_targets.csv")
        tci_daily = read_csv("tci_daily.csv")
        tci_quarterly = read_csv("tci_quarterly.csv")
        directional_spillover = read_csv("directional_spillover.csv")

    return {
        "spillover_targets": spillover_targets,
        "tci_daily": tci_daily,
        "tci_quarterly": tci_quarterly,
        "directional_spillover": directional_spillover,
    }

