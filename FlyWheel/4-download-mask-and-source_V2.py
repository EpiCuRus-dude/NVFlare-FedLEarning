#!/usr/bin/env python
""""""
import argparse
import logging
from pathlib import Path

from tqdm import tqdm
import flywheel


def get_dataview(client, project):

    builder = flywheel.ViewBuilder(
        label="DataView test",
        columns=[
            "subject.label",
            "session.label",
            "analysis.label",
            "analysis.id",
            "analysis.info.source_acq.label",
            "analysis.info.source_acq.id",
            "analysis.info.source_nifti.id",
            "analysis.info.source_nifti.name",
            "file.name",
        ],
        container="session",
        analysis_gear_name="roi2nix",
        filename="*.nii",
        match="all",  # 'newest'
        process_files=False,
        include_ids=False,
        include_labels=True,
        sort=False,
    )

    view = builder.build()
    view.parent = project.id
    view._error_column = None
    df = client.read_view_dataframe(view, project.id)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download mask and source NIfTI images."
    )
    parser.add_argument("--group", help="Group ID for the project", type=str)
    parser.add_argument("--project", help="Project label", type=str)
    parser.add_argument("--api-key", help="Flywheel API key", type=str)
    parser.add_argument(
        "--output", help="Output folder path where to download data", type=str
    )

    args = parser.parse_args()

    log = logging.getLogger()
    log.setLevel(logging.INFO)

    client = flywheel.Client(args.api_key)
    project = client.lookup(f"{args.group}/{args.project}")

    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    print(f'\n\n\n\noutput_dir={output_dir}\n\n\n')    

    df = get_dataview(client, project)

    for i, r in tqdm(df.iterrows()):
        log.info(f'Downloading {r["analysis.label"]}...')
        
         
        
        if not r["analysis.info.source_acq.label"]:
            log.warning(f'SKIPPING as as source NIfTI is not found')
            
            print(f'\t file.name={r["file.name"]}\n')
            
            log.warning(f'\t subject.label={r["subject.label"]}')
            log.warning(f'\t session.label={r["session.label"]}')
            #log.warning(f'\t acq.label={r["acquisition.label"]}')
            
            continue
            
        acq_path = (
            output_dir
            / r["subject.label"]
            / r["session.label"]
            / r["analysis.info.source_acq.label"]
        )
        if not acq_path.exists():
            acq_path.mkdir(parents=True)
            (acq_path / "mask").mkdir(parents=True)
            (acq_path / "source").mkdir(parents=True)

        # download source NIfTI
        acq = client.get_acquisition(r["analysis.info.source_acq.id"])
        acq.download_file(
            r["analysis.info.source_nifti.name"],
            str(acq_path / "source" / r["analysis.info.source_nifti.name"]),
        )

        # download mask NIfTI
        analysis = client.get_analysis(r["analysis.id"])
        
        
        ## ROI Not found!
        if r["file.name"] == None:
             continue
       
        
        
        
        print(f'\n\n\n **** \t file.name={r["file.name"]}\n')
        
        log.warning(f'\t subject.label={r["subject.label"]}')
        log.warning(f'\t session.label={r["session.label"]}')
        #log.warning(f'\t acq.label={r["acquisition.label"]}')
        
        
        
            
        analysis.download_file(
            r["file.name"],
            str(acq_path / "mask" / r["file.name"]),
        )
