"""
Profile-based strategy store.
Profiles are cross-symbol: one active profile is used for all stocks.
Create new profiles with 'finagent new-profile', switch with 'finagent use-profile'.
"""
from __future__ import annotations
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from finagent.config import PROFILES_DIR

_SEED_PATH = PROFILES_DIR / "default.json"


class ProfileNotFoundError(Exception):
    pass


class ProfileStore:
    def __init__(self, profiles_dir: Path = PROFILES_DIR):
        self.dir = profiles_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    # ── internal paths ──────────────────────────────────────────────────────

    def _profile_path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def _candidate_path(self, name: str) -> Path:
        return self.dir / f"{name}_candidate.json"

    def _active_file(self) -> Path:
        return self.dir / ".active"

    # ── active profile ───────────────────────────────────────────────────────

    def get_active_name(self) -> Optional[str]:
        """Return the active profile name, None if no profiles exist."""
        af = self._active_file()
        if af.exists():
            name = af.read_text(encoding="utf-8").strip()
            if name and self._profile_path(name).exists():
                return name
        # Fall back to most-recently-modified profile
        profiles = sorted(
            [p for p in self.dir.glob("*.json")
             if not p.name.endswith("_candidate.json")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return profiles[0].stem if profiles else None

    def set_active(self, name: str) -> None:
        if not self._profile_path(name).exists():
            raise ProfileNotFoundError(f"Profile '{name}' not found")
        self._active_file().write_text(name, encoding="utf-8")

    # ── profile CRUD ─────────────────────────────────────────────────────────

    def new_profile(self, name: Optional[str] = None) -> tuple:
        """
        Create a new profile from the default seed.
        Returns (name, path).
        Auto-generates a timestamped name if none given.
        """
        if name is None:
            name = "profile_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._profile_path(name)
        if path.exists():
            raise FileExistsError(f"Profile '{name}' already exists")

        seed = self._load_seed()
        seed.pop("symbol", None)
        seed.pop("symbol_specific_notes", None)
        seed.pop("strategy_version", None)
        seed["profile_name"] = name
        seed["profile_version"] = 0
        now = datetime.now(timezone.utc).isoformat()
        seed["created_at"] = now
        seed["updated_at"] = now
        seed["notes"] = ""
        seed["performance_history"] = {}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(seed, f, ensure_ascii=False, indent=2)
        return name, path

    def _load_seed(self) -> dict:
        if _SEED_PATH.exists():
            with open(_SEED_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {
            "predictor_system_prompt": "",
            "critic_system_prompt": "",
            "reflector_system_prompt": "",
            "evolver_system_prompt": "",
        }

    def load(self, name: Optional[str] = None) -> dict:
        """Load a profile by name, or the active profile if name is None."""
        if name is None:
            name = self.get_active_name()
        if name is None:
            # No profiles at all — bootstrap from seed
            name, _ = self.new_profile("default")
            self.set_active(name)
        path = self._profile_path(name)
        if not path.exists():
            raise ProfileNotFoundError(f"Profile '{name}' not found at {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_active(self) -> tuple:
        """Returns (name, profile_dict) for the active profile."""
        name = self.get_active_name()
        if name is None:
            name, _ = self.new_profile("default")
            self.set_active(name)
        return name, self.load(name)

    def save(self, name: str, profile: dict, as_candidate: bool = False) -> Path:
        """
        Persist an evolved profile.
        as_candidate=True → write {name}_candidate.json for human review.
        as_candidate=False → overwrite {name}.json directly.
        Increments profile_version.
        """
        profile = dict(profile)
        profile["profile_name"] = name
        profile.pop("symbol", None)
        profile["profile_version"] = profile.get("profile_version", 0) + 1
        profile["updated_at"] = datetime.now(timezone.utc).isoformat()

        path = self._candidate_path(name) if as_candidate else self._profile_path(name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        return path

    def patch(self, name: str, updates: dict) -> None:
        """
        Update specific fields in a profile JSON without incrementing profile_version.
        Useful for metadata updates (e.g., symbol_tags) that are not strategy evolutions.
        """
        path = self._profile_path(name)
        with open(path, encoding="utf-8") as f:
            profile = json.load(f)
        profile.update(updates)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    def promote_candidate(self, name: str) -> Path:
        """Rename candidate file → active profile file."""
        cand = self._candidate_path(name)
        active_path = self._profile_path(name)
        if not cand.exists():
            raise ProfileNotFoundError(f"No candidate for profile '{name}'")
        shutil.move(str(cand), str(active_path))
        return active_path

    # ── helpers ──────────────────────────────────────────────────────────────

    def patch_symbol_override(
        self, profile_name: str, symbol: str, override: dict, as_candidate: bool = False
    ) -> None:
        """
        Merge ``override`` into per_symbol_overrides[symbol] without incrementing profile_version.
        Used to store per-stock few-shot examples and calibration without polluting other stocks.
        as_candidate=True → patch the candidate file (used when store.save was also as_candidate).
        """
        path = self._candidate_path(profile_name) if as_candidate else self._profile_path(profile_name)
        if not path.exists():
            # Candidate may not exist yet if save hasn't been called; fall back to active
            path = self._profile_path(profile_name)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        overrides = data.setdefault("per_symbol_overrides", {})
        existing = overrides.get(symbol, {})
        existing.update(override)
        overrides[symbol] = existing
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_version(self, name: str) -> int:
        return self.load(name).get("profile_version", 0)

    def has_candidate(self, name: str) -> bool:
        return self._candidate_path(name).exists()

    def list_profiles(self) -> list:
        """Return metadata list for all profiles, newest-first."""
        active = self.get_active_name()
        result = []
        for p in sorted(
            [f for f in self.dir.glob("*.json")
             if not f.name.endswith("_candidate.json")],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        ):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                result.append({
                    "name": p.stem,
                    "version": data.get("profile_version", 0),
                    "updated_at": data.get("updated_at", ""),
                    "notes": data.get("notes", "")[:80],
                    "is_active": p.stem == active,
                    "has_candidate": self._candidate_path(p.stem).exists(),
                })
            except json.JSONDecodeError:
                pass
        return result
