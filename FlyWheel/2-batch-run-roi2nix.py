#!/usr/bin/env python
"""
A hierarchy-curator script that loop through all sessions and
triggers the roi2nix gear on annotated DICOM files.
"""
import argparse
import logging

import flywheel
from flywheel_gear_toolkit.utils.curator import HierarchyCurator
from flywheel_gear_toolkit.utils.walker import Walker


class Curator(HierarchyCurator):
    def __init__(self, *args, **kwargs):
        super(Curator, self).__init__(*args, **kwargs)
        self.roi2nix = self.client.lookup("gears/roi2nix")

    def curate_session(self, session):
        session = session.reload()

        ohif_blob = session.info.get("ohifViewer")
        if not ohif_blob:
            log.info(f"No OHIF annotation found for {session.ref()} - SKIPPING")
            return

        fhr = ohif_blob.get("measurements", {}).get("FreehandRoi")
        if not fhr:
            log.info(f"No OHIF FreehandROI found for {session.ref()} - SKIPPING")
            
            return

        suids = {ann.get("SeriesInstanceUID") for ann in fhr}
        
        #print(f'\n\n\n suids = {suids} \n\n\n')
        
        
        dicom_to_process = []
        for acq in session.acquisitions.iter():
            for f in acq.files:
                f = f.reload()
                if (
                    f.info.get("header", {}).get("dicom", {}).get("SeriesInstanceUID")
                    in suids
                ):
                    dicom_to_process.append(f)

        for dtp in dicom_to_process:
            inputs = {"Input_File": dtp}
            self.roi2nix.run(inputs=inputs, destination=session)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Batch run roi2nix gear on DICOM files that have annotations."
    )
    parser.add_argument("--group", help="Group ID for the project", type=str)
    parser.add_argument("--project", help="Project label", type=str)
    parser.add_argument("--api-key", help="Flywheel API key", type=str)

    args = parser.parse_args()

    log = logging.getLogger()
    log.setLevel(logging.INFO)

    client = flywheel.Client(args.api_key)
    project = client.lookup(f"{args.group}/{args.project}")

    curator = Curator(client=client)
    walker = Walker(project)
    for container in walker.walk():
        curator.curate_container(container)
