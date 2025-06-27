#!/usr/bin/env python3
"""
Automated release script for parametric-umap.

This script:
1. Checks git history for conventional commits
2. Determines version bump based on commit types
3. Updates version in pyproject.toml
4. Builds the package with uv
5. Publishes to PyPI
6. Creates and pushes git tag
7. Creates GitHub release

Usage:
    python scripts/release.py [--dry-run] [--verbose]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import tomlkit
from semantic_release import get_current_version, get_new_version
from semantic_release.commit_parser import parse_commit_message
from semantic_release.enums import LevelBump
from semantic_release.history import get_commits_since_version
from semantic_release.settings import config


def run_command(cmd: str, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"Error: {result.stderr.strip()}")
    else:
        result = subprocess.run(cmd, shell=True, check=check)
    
    return result


def get_git_commits_since_last_tag() -> list[str]:
    """Get all commits since the last git tag."""
    try:
        # Get the last tag
        result = run_command("git describe --tags --abbrev=0", check=False)
        if result.returncode == 0:
            last_tag = result.stdout.strip()
            # Get commits since last tag
            result = run_command(f"git log {last_tag}..HEAD --oneline --format='%s'")
        else:
            # No tags exist, get all commits
            result = run_command("git log --oneline --format='%s'")
        
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except Exception as e:
        print(f"Error getting git commits: {e}")
        return []


def analyze_commits_for_version_bump(commits: list[str]) -> LevelBump:
    """Analyze commits to determine version bump level."""
    has_breaking = False
    has_feature = False
    has_fix = False
    
    for commit in commits:
        commit = commit.strip()
        if not commit:
            continue
            
        # Check for breaking changes
        if "BREAKING CHANGE" in commit or commit.startswith("!"):
            has_breaking = True
        
        # Check conventional commit prefixes
        if commit.startswith(("feat:", "feat(")):
            has_feature = True
        elif commit.startswith(("fix:", "fix(", "perf:", "perf(")):
            has_fix = True
    
    # Determine bump level
    if has_breaking:
        return LevelBump.MAJOR
    elif has_feature:
        return LevelBump.MINOR
    elif has_fix:
        return LevelBump.PATCH
    else:
        return LevelBump.NO_RELEASE


def get_current_version_from_pyproject() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, "r") as f:
        data = tomlkit.load(f)
    
    return data["project"]["version"]


def update_version_in_pyproject(new_version: str) -> None:
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, "r") as f:
        data = tomlkit.load(f)
    
    data["project"]["version"] = new_version
    
    with open(pyproject_path, "w") as f:
        tomlkit.dump(data, f)
    
    print(f"Updated version in pyproject.toml to {new_version}")


def calculate_new_version(current_version: str, bump_level: LevelBump) -> str:
    """Calculate new version based on current version and bump level."""
    from packaging.version import Version
    
    version = Version(current_version)
    major, minor, patch = version.major, version.minor, version.micro
    
    if bump_level == LevelBump.MAJOR:
        return f"{major + 1}.0.0"
    elif bump_level == LevelBump.MINOR:
        return f"{major}.{minor + 1}.0"
    elif bump_level == LevelBump.PATCH:
        return f"{major}.{minor}.{patch + 1}"
    else:
        return current_version


def build_package() -> None:
    """Build the package using uv."""
    print("Building package...")
    run_command("uv build", capture_output=False)


def publish_to_pypi(dry_run: bool = False) -> None:
    """Publish package to PyPI using twine."""
    print("Publishing to PyPI...")
    
    cmd = "uv run twine upload dist/*"
    if dry_run:
        cmd += " --repository-url https://test.pypi.org/legacy/"
        print("DRY RUN: Would upload to Test PyPI")
    
    run_command(cmd, capture_output=False)


def create_git_tag(version: str, dry_run: bool = False) -> None:
    """Create and push git tag."""
    tag_name = f"v{version}"
    
    if dry_run:
        print(f"DRY RUN: Would create tag {tag_name}")
        return
    
    print(f"Creating git tag {tag_name}...")
    run_command(f"git tag -a {tag_name} -m 'Release {version}'")
    run_command(f"git push origin {tag_name}")


def create_github_release(version: str, dry_run: bool = False) -> None:
    """Create GitHub release using gh CLI."""
    tag_name = f"v{version}"
    
    if dry_run:
        print(f"DRY RUN: Would create GitHub release {tag_name}")
        return
    
    print(f"Creating GitHub release {tag_name}...")
    
    # Check if gh CLI is available
    try:
        run_command("gh --version", check=True)
    except subprocess.CalledProcessError:
        print("Warning: GitHub CLI (gh) not found. Skipping GitHub release creation.")
        print("Install gh CLI or create the release manually at: https://github.com/your-repo/releases")
        return
    
    # Create release notes
    release_notes = f"""# Release {version}

This release was automatically generated based on conventional commits.

## Changes
- See git commit history for detailed changes

🤖 Generated with automated release script
"""
    
    # Create the release
    run_command(f'gh release create {tag_name} --title "Release {version}" --notes "{release_notes}"')


def commit_version_changes(new_version: str, dry_run: bool = False) -> None:
    """Commit version changes to git."""
    if dry_run:
        print(f"DRY RUN: Would commit version bump to {new_version}")
        return
    
    print(f"Committing version bump to {new_version}...")
    run_command("git add pyproject.toml")
    run_command(f'git commit -m "chore: bump version to {new_version}"')
    run_command("git push origin")


def main():
    """Main release function."""
    parser = argparse.ArgumentParser(description="Automated release script")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("🚀 Starting automated release process...")
    
    # Check if we're on the main branch
    result = run_command("git branch --show-current")
    current_branch = result.stdout.strip()
    if current_branch != "main":
        print(f"Warning: You are on branch '{current_branch}', not 'main'")
        if not args.dry_run:
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
    
    # Get current version
    current_version = get_current_version_from_pyproject()
    print(f"Current version: {current_version}")
    
    # Get commits since last tag
    commits = get_git_commits_since_last_tag()
    if not commits or (len(commits) == 1 and not commits[0]):
        print("No new commits found. Nothing to release.")
        sys.exit(0)
    
    print(f"Found {len(commits)} commits to analyze:")
    for commit in commits[:10]:  # Show first 10
        print(f"  - {commit}")
    if len(commits) > 10:
        print(f"  ... and {len(commits) - 10} more")
    
    # Analyze commits for version bump
    bump_level = analyze_commits_for_version_bump(commits)
    
    if bump_level == LevelBump.NO_RELEASE:
        print("No version bump needed based on commit analysis.")
        print("Use conventional commits (feat:, fix:, etc.) to trigger releases.")
        sys.exit(0)
    
    # Calculate new version
    new_version = calculate_new_version(current_version, bump_level)
    print(f"Version bump: {current_version} -> {new_version} ({bump_level.name})")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes will be made:")
        print(f"  - Would update version to {new_version}")
        print("  - Would build package")
        print("  - Would publish to PyPI")
        print("  - Would create git tag and GitHub release")
        return
    
    # Confirm with user
    response = input(f"\nProceed with release {new_version}? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(1)
    
    try:
        # Update version
        update_version_in_pyproject(new_version)
        
        # Commit version changes
        commit_version_changes(new_version, args.dry_run)
        
        # Build package
        build_package()
        
        # Publish to PyPI
        publish_to_pypi(args.dry_run)
        
        # Create git tag
        create_git_tag(new_version, args.dry_run)
        
        # Create GitHub release
        create_github_release(new_version, args.dry_run)
        
        print(f"\n✅ Successfully released version {new_version}!")
        print(f"   - Package published to PyPI")
        print(f"   - Git tag v{new_version} created")
        print(f"   - GitHub release created")
        
    except Exception as e:
        print(f"\n❌ Release failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()