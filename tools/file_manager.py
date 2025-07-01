import os

def get_git_project_root(current_path: str) -> str:
    """Find the root path of current github project
    Args:
        str: currrent path of the
    Returns:
        str: string of the github project root path
    """
    root_indicators = ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt']
    while True:
        for indicator in root_indicators:
            if os.path.exists(os.path.join(current_path, indicator)):
                return current_path
            
        # Move up one directory
        parent_path = os.path.dirname(current_path)

        # If we reached the root of the filesystem, and no indicator was found
        if parent_path == current_path:
            return None  # Or raise an error if a root must be found
        current_path = parent_path

