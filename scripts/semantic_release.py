#!/usr/bin/env python3
"""
Simplified semantic release script using python-semantic-release.

This script provides a streamlined interface to the python-semantic-release tool
configured in pyproject.toml.

Usage:
    python scripts/semantic_release.py [--dry-run] [--verbose]
    python scripts/semantic_release.py version  # Show next version
    python scripts/semantic_release.py changelog  # Generate changelog
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, text=True)
    return result


def check_dependencies():
    """Check that required tools are available."""
    tools = ["uv", "git"]
    missing = []
    
    for tool in tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(tool)
    
    if missing:
        print(f"❌ Missing required tools: {', '.join(missing)}")
        sys.exit(1)


def show_next_version():
    """Show what the next version would be."""
    print("🔍 Analyzing commits to determine next version...")
    result = run_command("uv run semantic-release version --print", check=False)
    if result.returncode != 0:
        print("No version bump needed or unable to determine version.")
    return result.returncode == 0


def generate_changelog():
    """Generate changelog."""
    print("📝 Generating changelog...")
    run_command("uv run semantic-release changelog")


def do_release(dry_run: bool = False, verbose: bool = False):
    """Perform the full release process."""
    print("🚀 Starting semantic release process...")
    
    # Check if we're on the main branch
    result = subprocess.run(
        ["git", "branch", "--show-current"], 
        capture_output=True, text=True, check=True
    )
    current_branch = result.stdout.strip()
    
    if current_branch != "main":
        print(f"Warning: You are on branch '{current_branch}', not 'main'")
        if not dry_run:
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
    
    # Build the semantic-release command
    cmd = "uv run semantic-release version"
    
    if dry_run:
        cmd += " --print"
        print("🔍 DRY RUN - Showing what would be released:")
    
    if verbose:
        cmd += " --verbose"
    
    # Run semantic-release
    result = run_command(cmd, check=False)
    
    if result.returncode != 0:
        if "No release will be made" in result.stderr or not result.stderr:
            print("✨ No release needed - no conventional commits found since last release.")
        else:
            print(f"❌ Semantic release failed: {result.stderr}")
        sys.exit(result.returncode)
    
    if not dry_run:
        print("✅ Release completed successfully!")
        print("📦 Package should now be available on PyPI")
        print("🏷️  Git tag and GitHub release created")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Semantic release for parametric-umap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/semantic_release.py                    # Perform full release
  python scripts/semantic_release.py --dry-run          # Show what would be released
  python scripts/semantic_release.py version            # Show next version
  python scripts/semantic_release.py changelog          # Generate changelog
        """
    )
    
    parser.add_argument(
        "command", 
        nargs="?", 
        choices=["version", "changelog"], 
        help="Command to run (default: full release)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Execute based on command
    if args.command == "version":
        show_next_version()
    elif args.command == "changelog":
        generate_changelog()
    else:
        # Full release
        do_release(args.dry_run, args.verbose)


if __name__ == "__main__":
    main()