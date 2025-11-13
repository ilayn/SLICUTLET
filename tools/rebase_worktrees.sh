#!/bin/bash
# Rebase all worktrees before merging
# Usage: ./tools/rebase_worktrees.sh wt1-branch wt2-branch wt3-branch

set -e  # Exit on error

MAIN_DIR="/Users/josephj/Workspace/SLICUTLET"
BASE_DIR="/Users/josephj/Workspace"

if [ $# -eq 0 ]; then
    echo "Usage: $0 branch1 branch2 [branch3 ...]"
    echo "Example: $0 wt1-ma01ad wt2-ma01bd wt3-ma02bd"
    exit 1
fi

echo "🔄 Rebasing all worktrees on latest main..."
echo ""

for branch in "$@"; do
    # Extract worktree number (e.g., wt1 from wt1-mb04od)
    wt_num=$(echo "$branch" | grep -o 'wt[0-9]\+')

    if [ -z "$wt_num" ]; then
        echo "⚠️  WARNING: Cannot extract worktree number from '$branch', skipping..."
        continue
    fi

    wt_dir="$BASE_DIR/SLICUTLET-$wt_num"

    if [ -d "$wt_dir" ]; then
        echo "📦 Rebasing $branch in $wt_dir..."
        cd "$wt_dir" || exit 1

        # Fetch latest main from main worktree
        git fetch "$MAIN_DIR" main:main

        # Rebase on main
        git rebase main

        if [ $? -ne 0 ]; then
            echo "❌ ERROR: Rebase failed for $branch"
            echo "   Please resolve conflicts manually and run:"
            echo "   cd $wt_dir && git rebase --continue"
            git rebase --abort
            exit 1
        fi

        echo "✅ Successfully rebased $branch"
        echo ""
    else
        echo "⚠️  WARNING: Worktree directory $wt_dir not found, skipping..."
        echo ""
    fi
done

cd "$MAIN_DIR" || exit 1
echo "🎉 All worktrees rebased successfully!"
echo ""
echo "Ready to merge branches:"
for branch in "$@"; do
    echo "  - $branch"
done
echo ""
echo "Run from main worktree:"
echo "  cd $MAIN_DIR"
for branch in "$@"; do
    echo "  git merge $branch"
done
