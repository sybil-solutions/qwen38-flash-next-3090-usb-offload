#!/bin/sh
# Render exl3-card.html at 1:1 (1200x675) with headless Brave into ../exl3-card.png (same method as local-omarchy/brand/export.sh).
# The page loads the brand scripts from ../../local-omarchy/brand: $BRAND (default $HOME/local-omarchy/brand) is linked there in a staging directory,
# so the card renders from any checkout of this repo.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BRAND="${BRAND:-$HOME/local-omarchy/brand}"
[ -f "$BRAND/brand.js" ] && [ -f "$BRAND/hands.js" ] || { echo "no brand scripts in $BRAND (set BRAND=<dir with brand.js and hands.js>)" >&2; exit 1; }
S="$(mktemp -d)"; trap 'rm -rf "$S"' EXIT
mkdir -p "$S/docs/card" "$S/local-omarchy"
cp "$HERE/card.js" "$HERE/exl3-card.html" "$S/docs/card/"
ln -s "$BRAND" "$S/local-omarchy/brand"
B="/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
perl -e 'alarm 60; exec @ARGV' "$B" --headless=new --hide-scrollbars --disable-gpu --force-device-scale-factor=1 \
  --allow-file-access-from-files --virtual-time-budget=6000 --window-size=1200,675 --screenshot="$S/docs/card/exl3-card.png" \
  "file://$S/docs/card/exl3-card.html" >/dev/null 2>&1
cp "$S/docs/card/exl3-card.png" "$HERE/../exl3-card.png"
ls -la "$HERE/../exl3-card.png"
