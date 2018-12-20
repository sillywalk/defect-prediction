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
    def __init__(self, level="file"):
        """
        A Generic data handler class

        Parameters
        ----------
        data_path = <pathlib.PosixPath>
            Path to the data folder.
        """
        data_path = root.joinpath("data/" + level + "_level")
        print(data_path)
        self.level = level
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
        #print(types)
        if self.level == "file":
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
                                if i >= len(versions) // 3:
                                    break
                                ver = "%s_%s_file_metrics_ready.csv" % (p.name, i)
                            temp_df.append(pd.read_csv(t.joinpath(p, ver)))
                        temp_dict.update(OrderedDict({p.name: temp_df}))
                    all_data[t.name] = temp_dict
        else:
            label_types = ['keyword', 'fastread', 'human']
            for label_t in label_types:
                temp_dict = OrderedDict()
                irr_columns = [l + "_buggy_fixes" for l in label_types if l != label_t]
                #irr_columns.append("time")
                for p in types:
                    versions = [filename for filename in os.listdir(str(p)) if filename.endswith(".csv")]
                    temp_df = []
                    for i in range(len(versions)):
                        ver = "%s_%s.csv" % (p.name, i + 1)
                        temp_ver_df = pd.read_csv(self.data_path.joinpath(p, ver))
                        temp_ver_df = temp_ver_df.drop(irr_columns, axis=1)
                        temp_df.append(temp_ver_df)
                    temp_dict.update(OrderedDict({p.name: temp_df}))
                all_data[label_t] = temp_dict
        return all_data


if __name__ == "__main__":
    dh = DataHandler('commit')
    dh.get_data()