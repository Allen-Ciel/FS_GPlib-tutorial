import argparse
import os
from pathlib import Path

import polib

import argostranslate.package
import argostranslate.translate


def ensure_argos_model(from_code: str = "en", to_code: str = "zh") -> None:
    """
    Ensure that the Argos Translate model for from_code -> to_code is installed.
    If not installed, download and install it automatically.
    """
    # Check if already installed
    installed_packages = argostranslate.package.get_installed_packages()
    for pkg in installed_packages:
        if getattr(pkg, "from_code", None) == from_code and getattr(pkg, "to_code", None) == to_code:
            return

    # Update index and install the matching package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        (
            p
            for p in available_packages
            if getattr(p, "from_code", None) == from_code and getattr(p, "to_code", None) == to_code
        ),
        None,
    )

    if package_to_install is None:
        raise RuntimeError(f"No Argos Translate package found for {from_code} -> {to_code}")

    download_path = package_to_install.download()
    argostranslate.package.install_from_path(download_path)


def translate_text(text: str, from_code: str = "en", to_code: str = "zh") -> str:
    """
    Translate a single string using Argos Translate.
    """
    if not text.strip():
        return text
    return argostranslate.translate.translate(text, from_code, to_code)


def auto_translate_po_files(
    locale_root: Path,
    lang: str = "zh_CN",
    source_lang: str = "en",
    target_lang_for_engine: str = "zh",
    overwrite: bool = False,
) -> None:
    """
    Walk through all .po files under locale_root/<lang>/LC_MESSAGES and
    fill msgstr using Argos Translate.
    """
    lang_dir = locale_root / lang / "LC_MESSAGES"
    if not lang_dir.exists():
        raise SystemExit(f"Locale directory not found: {lang_dir}")

    ensure_argos_model(source_lang, target_lang_for_engine)

    po_files = list(lang_dir.rglob("*.po"))
    if not po_files:
        print(f"No .po files found under {lang_dir}")
        return

    print(f"Translating {len(po_files)} .po files under {lang_dir} ...")

    for po_path in po_files:
        _translate_single_po_file(
            po_path=po_path,
            source_lang=source_lang,
            target_lang_for_engine=target_lang_for_engine,
            overwrite=overwrite,
        )


def _translate_single_po_file(
    po_path: Path,
    source_lang: str,
    target_lang_for_engine: str,
    overwrite: bool,
) -> None:
    """
    Translate a single .po file in place.
    """
    print(f"Processing {po_path} ...")
    po = polib.pofile(str(po_path))
    changed = False

    for entry in po:
        if entry.obsolete:
            continue

        # Skip already translated entries unless overwrite is requested
        if not overwrite and entry.msgstr:
            continue

        src = entry.msgid.strip()
        if not src:
            continue

        try:
            translated = translate_text(src, from_code=source_lang, to_code=target_lang_for_engine)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Failed to translate '{src}': {exc}")
            continue

        if translated and translated != entry.msgstr:
            entry.msgstr = translated
            # Clear fuzzy flag when we programmatically translate
            if "fuzzy" in entry.flags:
                entry.flags.remove("fuzzy")
            changed = True

    if changed:
        po.save()
        print(f"  Saved updated translations to {po_path}")
    else:
        print(f"  No changes for {po_path}")


def auto_translate_specific_po_files(
    po_files: list[Path],
    source_lang: str = "en",
    target_lang_for_engine: str = "zh",
    overwrite: bool = False,
) -> None:
    """
    Translate only the specified .po files.
    """
    if not po_files:
        print("No .po files specified.")
        return

    ensure_argos_model(source_lang, target_lang_for_engine)

    print(f"Translating {len(po_files)} specified .po files ...")

    for po_path in po_files:
        print(f"Processing {po_path} ...")
        _translate_single_po_file(
            po_path=po_path,
            source_lang=source_lang,
            target_lang_for_engine=target_lang_for_engine,
            overwrite=overwrite,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically machine-translate Sphinx .po files using Argos Translate.\n\n"
            "Typical usage (from docs/ directory):\n"
            "  python auto_translate_po.py --lang zh_CN\n"
        )
    )
    parser.add_argument(
        "--lang",
        default="zh_CN",
        help="Sphinx locale language code to translate into (default: zh_CN).",
    )
    parser.add_argument(
        "--source-lang",
        default="en",
        help="Source language code (default: en).",
    )
    parser.add_argument(
        "--target-lang-engine",
        default="zh",
        help="Target language code used by Argos Translate engine (default: zh).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing msgstr values instead of only filling empty ones.",
    )
    parser.add_argument(
        "--locale-root",
        default="source/locale",
        help="Root directory of Sphinx locale files (default: source/locale).",
    )
    parser.add_argument(
        "--po",
        action="append",
        help=(
            "Specific .po file to translate (relative or absolute path). "
            "Can be given multiple times; if provided, only these files are translated."
        ),
    )

    args = parser.parse_args()

    # If specific .po files are provided, only translate those
    if args.po:
        po_paths = [Path(p).resolve() for p in args.po]
        auto_translate_specific_po_files(
            po_files=po_paths,
            source_lang=args.source_lang,
            target_lang_for_engine=args.target_lang_engine,
            overwrite=bool(args.overwrite),
        )
    else:
        cwd = Path(os.getcwd())
        locale_root = (cwd / args.locale_root).resolve()

        auto_translate_po_files(
            locale_root=locale_root,
            lang=args.lang,
            source_lang=args.source_lang,
            target_lang_for_engine=args.target_lang_engine,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()

