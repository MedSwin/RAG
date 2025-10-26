import os 
from pathlib import Path
import json
import pandas as pd           
from typing import Dict, List 
import hashlib
from datetime import datetime
                    
# def load_data(directory: str) -> pd.DataFrame:
#     print(f"Load document from the directory: {directory}")
#     list_df = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".csv"):
#             try:    
#                 dff = pd.read_csv(os.path.join(directory, filename))
#                 list_df.append(dff)
#                 print(f"Loaded {filename} with {len(dff)} rows into")
#             except Exception as e:
#                 print(f"Error loading {filename}: {e}")
    
#     if list_df:
#         df = pd.concat(list_df, ignore_index=True)
#         print(f"Total combined dataframe: {len(df)} rows")
#         return df
#     else:
#         return pd.DataFrame()
        

class DocumentIngestionManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.state_file = self.base_dir / "ingestion_state.json"
        self.processed_docs = self.load_state()
        
    def load_state(self) -> dict:
        """Load the state of processed documents

        Returns:
            dict: dict of processed documents
        """
        
        try:
            with open(self.state_file, "r") as file:
               return json.load(file)
        except Exception as e:
            print(f"Can not load state as error {e}")
            return {}
    
    def save_state(self) -> None:
        """Save the state of processed documents

        Returns:
            dict: dict of processed documents updated
        """
        
        try:
            with open(self.state_file, "w") as file:
                json.dump(self.processed_docs, file, indent=2)
        except Exception as e:
            print(f"Can not save the file due to error: {e}")
            
    def hash_file(self, file_path: Path) -> str:
        """ Hash the file

        Args:
            file_path (Path): file need to be hashed
        """
        with open(file_path, 'rb') as file:
            content = file.read()
            return hashlib.md5(content).hexdigest()
        
            
    def check_new_or_modified_files(self, directory: str) -> List[Path]:
        """Check for new or modified files in the directory

        Args:
            directory (str): directory to check

        Returns:
            List[Path]: list of new or modified files
        """
        directory = Path(directory)
        new_or_modified = []
        
        for file_path in directory.glob("**/*.csv"):
            file_key = str(file_path.relative_to(directory))
            current_hash = self.hash_file(file_path)
            current_mtime = file_path.stat().st_mtime
            
            if file_key not in self.processed_docs.keys():
                new_or_modified.append(file_path)
                print(f"New file detected: {file_key}")
            elif ((self.processed_docs[file_key].get("hash") != current_hash)
                  or (self.processed_docs[file_key].get("mtime") != current_mtime)):
                new_or_modified.append(file_path)
                print(f"Modified file detected: {file_path}")
        
        return new_or_modified
    
    def marked_as_processed(self, file_path: Path, directory: Path) -> None:
        """Mark a processed file as processed to avoid re-process

        Args:
            file_path (Path): path of the file
            directory (Path): directory that contains the data file
        """
        file_key = str(file_path.relative_to(directory))
        self.processed_docs[file_key] = {
            "hash": self.hash_file(file_path),
            "mtime": file_path.stat().st_mtime,
            "processed_aat": datetime.now().isoformat()
        }
        
    def incremental_load(self, directory: str) -> pd.DataFrame:
        """Just load new file or modified file

        Args:
            directory (str): directory of all the data

        Returns:
            pd.DataFrame: data frame
        """
        directory = Path(directory)
        new_files = self.check_new_or_modified_files(directory)
        
        if not new_files:
            print("No new file found")
            return pd.DataFrame()
        
        df = []
        for file_path in new_files:
            try:
                dfs = pd.read_csv(file_path)
                df.append(dfs)
                print(f"Added {file_path} to data frame")
                
                self.marked_as_processed(file_path, directory)
                print(f"Processed {file_path}")
                
            except Exception as e:
                print(f"Can not load {file_path} into data frame")
                
        self.save_state()
        
        return pd.concat(df, ignore_index=True) if df else pd.DataFrame()
        
                
        
    
if __name__ == "__main__":
    manager = DocumentIngestionManager("/fred/oz446/HenryNguyen/data")
    
    initial_data = manager.incremental_load("/fred/oz446/HenryNguyen/data")
    
    print(initial_data.head(5))