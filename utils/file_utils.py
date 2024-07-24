import os
from pathlib import Path

def get_filenames(folder_path):
  """Gets a list of filenames in a given folder.

  Args:
    folder_path: The path to the folder.

  Returns:
    A list of filenames inside a folder.
  """

  filenames = []
  for root, dirs, files in os.walk(folder_path):
    for file in files:
      filenames.append(file)
  return filenames