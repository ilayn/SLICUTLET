#!/usr/bin/env python3
"""
Extract dependency tree from SLICOT Fortran 77 source files.

This tool parses Fortran 77 source files to identify:
1. SLICOT routine dependencies (calls to other SLICOT routines)
2. LAPACK/BLAS dependencies (for FFI planning)
3. Dependency levels (leaf routines vs. higher-level routines)

Usage:
    python extract_dependencies.py <fortran_source_dir>
    python extract_dependencies.py reference/src/
    python extract_dependencies.py reference/src/AB01MD.f  # Single file
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple


class FortranDependencyAnalyzer:
    """Analyzes Fortran 77 source files to extract routine dependencies."""

    # Regex patterns
    SLICOT_CALL_PATTERN = re.compile(r'^\s+CALL\s+([A-Z]{2}[0-9]{2}[A-Z]{2})\s*\(', re.IGNORECASE)
    LAPACK_CALL_PATTERN = re.compile(r'^\s+CALL\s+(D[A-Z]{3,5}|XERBLA|ILAENV)\s*\(', re.IGNORECASE)
    ROUTINE_NAME_PATTERN = re.compile(r'^\s+SUBROUTINE\s+([A-Z]{2}[0-9]{2}[A-Z]{2})', re.IGNORECASE)

    def __init__(self):
        self.slicot_deps: Dict[str, Set[str]] = defaultdict(set)  # routine -> SLICOT dependencies
        self.lapack_deps: Dict[str, Set[str]] = defaultdict(set)  # routine -> LAPACK/BLAS dependencies
        self.all_routines: Set[str] = set()

    def parse_file(self, filepath: Path) -> None:
        """Parse a single Fortran file to extract dependencies."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.readlines()
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
            return

        current_routine = None

        # First pass: identify routine name from SUBROUTINE statement
        for line in content:
            match = self.ROUTINE_NAME_PATTERN.match(line)
            if match:
                current_routine = match.group(1).upper()
                self.all_routines.add(current_routine)
                break

        # If no SUBROUTINE found, try to extract from filename
        if not current_routine:
            filename = filepath.stem.upper()
            if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}$', filename):
                current_routine = filename
                self.all_routines.add(current_routine)

        if not current_routine:
            return  # Not a SLICOT routine file

        # Second pass: extract CALL statements
        for line in content:
            # Check for SLICOT calls
            slicot_match = self.SLICOT_CALL_PATTERN.match(line)
            if slicot_match:
                called_routine = slicot_match.group(1).upper()
                self.slicot_deps[current_routine].add(called_routine)

            # Check for LAPACK/BLAS calls
            lapack_match = self.LAPACK_CALL_PATTERN.match(line)
            if lapack_match:
                called_routine = lapack_match.group(1).upper()
                # Filter out error handling
                if called_routine != 'XERBLA' and called_routine != 'ILAENV':
                    self.lapack_deps[current_routine].add(called_routine)

    def parse_directory(self, directory: Path) -> None:
        """Parse all .f files in a directory."""
        fortran_files = sorted(directory.glob('*.f'))
        print(f"Parsing {len(fortran_files)} Fortran files from {directory}...")

        for filepath in fortran_files:
            self.parse_file(filepath)

        print(f"Found {len(self.all_routines)} SLICOT routines")

    def compute_dependency_levels(self) -> Dict[str, int]:
        """
        Compute dependency levels for all routines.
        Level 0: No SLICOT dependencies (leaf routines)
        Level 1: Depends only on Level 0
        Level N: Depends on routines up to Level N-1
        """
        levels = {}
        remaining = set(self.all_routines)
        current_level = 0

        while remaining:
            # Find routines whose all dependencies are already assigned levels
            level_routines = set()

            for routine in remaining:
                deps = self.slicot_deps.get(routine, set())

                if not deps:  # No SLICOT dependencies
                    level_routines.add(routine)
                else:
                    # Check if all dependencies have assigned levels
                    deps_resolved = all(dep in levels or dep not in self.all_routines
                                       for dep in deps)
                    if deps_resolved:
                        # Check maximum level of dependencies
                        dep_levels = [levels.get(dep, -1) for dep in deps if dep in levels]
                        if not dep_levels or max(dep_levels) < current_level:
                            level_routines.add(routine)

            if not level_routines:
                # Handle circular dependencies or external dependencies
                for routine in remaining:
                    levels[routine] = current_level
                break

            for routine in level_routines:
                levels[routine] = current_level
                remaining.remove(routine)

            current_level += 1

        return levels

    def print_summary(self) -> None:
        """Print dependency analysis summary."""
        print("\n" + "="*80)
        print("DEPENDENCY ANALYSIS SUMMARY")
        print("="*80)

        levels = self.compute_dependency_levels()

        # Group by level
        by_level: Dict[int, List[str]] = defaultdict(list)
        for routine, level in levels.items():
            by_level[level].append(routine)

        # Print level statistics
        print(f"\nTotal routines: {len(self.all_routines)}")
        print(f"Dependency levels: {len(by_level)}")
        print()

        for level in sorted(by_level.keys()):
            routines = sorted(by_level[level])
            print(f"Level {level}: {len(routines)} routines")
            if level == 0:
                print("  (Leaf routines - no SLICOT dependencies, only BLAS/LAPACK)")

        print("\n" + "-"*80)
        print("LEVEL 0 ROUTINES (Translation Priority - can be done in parallel)")
        print("-"*80)

        leaf_routines = sorted(by_level[0])
        for routine in leaf_routines[:50]:  # Show first 50
            lapack = sorted(self.lapack_deps.get(routine, set()))
            if lapack:
                print(f"  {routine:8s} → LAPACK: {', '.join(lapack)}")
            else:
                print(f"  {routine:8s} → No external dependencies (pure Fortran)")

        if len(leaf_routines) > 50:
            print(f"  ... and {len(leaf_routines) - 50} more")

        print("\n" + "-"*80)
        print("HIGHER-LEVEL ROUTINES (sample)")
        print("-"*80)

        for level in sorted(by_level.keys())[1:4]:  # Show levels 1-3
            routines = sorted(by_level[level])[:10]  # First 10 per level
            print(f"\nLevel {level} (sample):")
            for routine in routines:
                deps = sorted(self.slicot_deps.get(routine, set()))
                lapack = sorted(self.lapack_deps.get(routine, set()))
                print(f"  {routine:8s} → SLICOT: {', '.join(deps[:5])}")
                if len(deps) > 5:
                    print(f"            (+ {len(deps) - 5} more)")
                if lapack:
                    print(f"            LAPACK: {', '.join(lapack[:5])}")

    def print_routine_detail(self, routine_name: str) -> None:
        """Print detailed dependency information for a specific routine."""
        routine_name = routine_name.upper()

        if routine_name not in self.all_routines:
            print(f"Error: Routine {routine_name} not found")
            return

        levels = self.compute_dependency_levels()
        level = levels.get(routine_name, -1)

        print("\n" + "="*80)
        print(f"ROUTINE: {routine_name}")
        print("="*80)
        print(f"Dependency Level: {level}")

        # SLICOT dependencies
        slicot_deps = sorted(self.slicot_deps.get(routine_name, set()))
        if slicot_deps:
            print(f"\nSLICOT Dependencies ({len(slicot_deps)}):")
            for dep in slicot_deps:
                dep_level = levels.get(dep, '?')
                implemented = dep in self.all_routines
                status = "✓" if implemented else "✗"
                print(f"  {status} {dep:8s} (Level {dep_level})")
        else:
            print("\nNo SLICOT dependencies (LEAF ROUTINE)")

        # LAPACK dependencies
        lapack_deps = sorted(self.lapack_deps.get(routine_name, set()))
        if lapack_deps:
            print(f"\nLAPACK/BLAS Dependencies ({len(lapack_deps)}):")
            for dep in lapack_deps:
                print(f"  {dep}")

        # Reverse dependencies (who depends on this?)
        reverse_deps = [r for r in self.all_routines
                       if routine_name in self.slicot_deps.get(r, set())]
        if reverse_deps:
            print(f"\nUsed By ({len(reverse_deps)} routines):")
            for dep in sorted(reverse_deps)[:20]:
                print(f"  {dep}")
            if len(reverse_deps) > 20:
                print(f"  ... and {len(reverse_deps) - 20} more")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_dependencies.py <fortran_source_dir|file>")
        print("   or: python extract_dependencies.py <fortran_source_dir> <routine_name>")
        sys.exit(1)

    source_path = Path(sys.argv[1])
    analyzer = FortranDependencyAnalyzer()

    if source_path.is_file():
        analyzer.parse_file(source_path)
    elif source_path.is_dir():
        analyzer.parse_directory(source_path)
    else:
        print(f"Error: {source_path} is not a valid file or directory")
        sys.exit(1)

    if len(sys.argv) == 3:
        # Show specific routine details
        routine_name = sys.argv[2].upper()
        analyzer.print_routine_detail(routine_name)
    else:
        # Show summary
        analyzer.print_summary()


if __name__ == '__main__':
    main()
