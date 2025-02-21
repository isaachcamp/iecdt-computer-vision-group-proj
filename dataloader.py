import xarray as xr
from torch.utils.data import DataLoader
import pandas as pd

class CloudLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        DataLoader class to read...
        """
        super(CloudLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def __getitem__(self, path):
        """
        Get the item at the given index.
        """
        return self.df['path'] == path




    def load_data(self):
        """
        Load the NetCDF file using xarray and store the dataset.
        """
        try:
            self.dataset = xr.open_dataset(self.file_path)
            print(f"Successfully loaded dataset from {self.file_path}")
        except Exception as e:
            print(f"Failed to load dataset: {e}")

    def get_dataset(self):
        """
        Get the loaded dataset.

        Returns:
        xarray.Dataset: The loaded dataset.
        """
        return self.dataset
    

#    def convert_time_to_unix(self):
        """
        Convert the 'time' coordinate of the xarray dataset from datetime format to unix format.
        """
        if self.dataset is not None:
            try:
                self.dataset['time'] = self.dataset['time'].astype('datetime64[s]').astype(int)
                print("Successfully converted 'time' coordinate to unix format.")
            except Exception as e:
                print(f"Failed to convert 'time' coordinate: {e}")
        else:
            print("Dataset is not loaded. Please load the dataset first.")


# Example usage:
# dataloader = DataLoader('/path/to/your/netcdf/file.nc')
# dataloader.load_data()
# dataset = dataloader.get_dataset()
# print(dataset)