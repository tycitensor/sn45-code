import os
import shutil
import tempfile
import weakref
from git import Repo

class GitRepo:
    def __init__(self, repo_name: str, commit_hash: str):
        """
        Initialize a Git repository object that manages cloning and cleanup.
        
        Args:
            repo_name (str): Name/URL of the repository to clone
            commit_hash (str): Specific commit hash to checkout
        """
        self.repo_name = repo_name
        self.commit_hash = commit_hash
        self.temp_dir = tempfile.mkdtemp()
        
        # Clone repo and checkout specific commit
        self.repo = Repo.clone_from(self.repo_name, self.temp_dir)
        self.repo.git.checkout(self.commit_hash)
        
        # Register cleanup to be called when object is deleted
        self._finalizer = weakref.finalize(self, self._cleanup)
        
    def _cleanup(self):
        """
        Clean up the temporary directory containing the cloned repository.
        """
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            
    @property
    def path(self) -> str:
        """
        Get the path to the cloned repository.
        
        Returns:
            str: Path to the repository directory
        """
        return self.temp_dir
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
