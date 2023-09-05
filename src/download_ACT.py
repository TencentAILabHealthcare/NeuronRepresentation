import pandas as pd
import os
import wget


def download_allen_cell_type_data(
    folder_output, fname_csv="./data/info/cell_types_specimen_details.csv"
):
    request_url = (
        "https://celltypes.brain-map.org/api/v2/well_known_file_download/{nrwkf__id}"
    )

    df = pd.read_csv(fname_csv)
    df = df[df["nrwkf__id"].notnull()]
    df = df[df["donor__species"] == "Mus musculus"]

    for idx, row in df.iterrows():
        dataid = row["nrwkf__id"]
        url = request_url.replace("{nrwkf__id}", str(dataid))
        fname = wget.download(url, out=folder_output)
        fname = os.path.basename(fname)
        print("\n", int(idx), fname)
        df.loc[idx, "swc__fname"] = fname

    df.to_csv(os.path.join(os.path.dirname(fname_csv), "ACT_info_swc.csv"))


if __name__ == "__main__":
    print("Download swc file from AIBS")
    os.makedirs("./data/raw/allen_cell_type/swc")
    download_allen_cell_type_data(os.path.join("./data", "raw/allen_cell_type/swc"))
