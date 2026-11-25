#!/bin/zsh
# backup.sh: back the dataset up, for a nightly job.
#   1. A consistent copy of the database (label history included) beside it, in backups/, newest 30 kept.
#   2. A snapshot (`adbreak dataset export`) committed to the `dataset-snapshots` branch and pushed, only when the
#      dataset changed: a copy off this machine, with history.
# Needs `adbreak` on PATH (or ADBREAK_BIN) and push access to the repository's origin.
set -eu
repo=${ADBREAK_REPO:-${0:A:h:h}}
adbreak=${ADBREAK_BIN:-adbreak}
branch=dataset-snapshots
wt=$repo/.worktrees/$branch

echo "$(date '+%F %T') backup: $($adbreak dataset backup)"

cd $repo
git fetch -q origin
if [ ! -d $wt ]; then
  if git ls-remote --exit-code --heads origin $branch >/dev/null; then
    git worktree add -q $wt -B $branch origin/$branch
  else
    git worktree add -q --orphan -b $branch $wt
  fi
fi
cd $wt
git pull -q --ff-only origin $branch 2>/dev/null || true
rm -rf dataset
n=$($adbreak dataset export dataset 2>&1 | grep -o '[0-9]* recordings' || echo "? recordings")
git add -A dataset
if git diff --cached --quiet; then
  echo "$(date '+%F %T') snapshot: no changes"
  exit 0
fi
git commit -q -m "Dataset snapshot $(date -u '+%F %H:%M') UTC: $n"
git push -q origin $branch
echo "$(date '+%F %T') snapshot: pushed $(git rev-parse --short HEAD) ($n)"
