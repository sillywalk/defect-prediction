"""
A data handler to read, write, and process data
"""
import os
import sys
import pandas as pd
from glob2 import glob
from pdb import set_trace
from collections import OrderedDict

from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))

if root not in sys.path:
    sys.path.append(str(root))


class DataHandler:
    def __init__(self, data_path=root.joinpath("data")):
        """
        A Generic data handler class

        Parameters
        ----------
        data_path = <pathlib.PosixPath>
            Path to the data folder.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Read data as pandas and return a dictionary of data

        Returns
        -------
        all_data: dict
            A dictionary of data with key-project_name, value-list of file
            level metrics
        """

        all_data = OrderedDict()
        types = [Path(t) for t in glob(str(self.data_path.joinpath("[!_]*"))) if Path(t).is_dir()]
        print(types)
        for t in types:
            if t.name != "keyword_guru":
                projects = [Path(p) for p in glob(str(t.joinpath("[!_]*"))) if Path(p).is_dir()]
                temp_dict = OrderedDict()
                for p in projects:
                    versions = [filename for filename in os.listdir(str(p)) if filename.endswith(".csv")]
                    temp_df = []
                    for i in range(len(versions)):
                        if t.name != "release_level":
                            ver = "%s_%s_%s_ready.csv" % (p.name, i, t.name)
                        else:
                            #print(t, p, i)
                            if i >= len(versions) // 2:
                                break
                            ver = "%s_%s_ready.csv" % (p.name, i)
                        temp_df.append(pd.read_csv(t.joinpath(p, ver)))
                    temp_dict.update(OrderedDict({p.name: temp_df}))
                all_data[t.name] = temp_dict
        #print(all_data.keys())
        #set_trace()
        return all_data
