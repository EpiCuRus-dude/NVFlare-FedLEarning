#!/usr/bin/env python
"""
A script that matches source nifti files for roi2nix analysis container.
"""
import argparse
import logging

import flywheel
from flywheel_gear_toolkit.utils.curator import HierarchyCurator
from flywheel_gear_toolkit.utils.walker import Walker


class Curator(HierarchyCurator):
    def curate_session(self, session):
        """Tag analysis nifti mask and source nifti"""
        session = session.reload()
        print(f'session.label = {session.label}\n******\t')
        analyses = session.analyses
        for ana in analyses:
            print(f'ana.id={ana.id}')
            print(f'ana.gear_info={ana.gear_info}\n\n')
            if ana.gear_info == None:
                continue
                
            if ana.gear_info.name == "roi2nix" and ana.job.state == "complete":
                log.info(f"Tagging analysis {ana.label} {session.label} {ana.ref()}...")
                # tagging the corresponding nifti file
                in_file_ref = ana.inputs[0].ref()
                acq = client.get_acquisition(ana.inputs[0].parents.acquisition)
                source_nifti = None
                for f in acq.files:
                    if f.type == "nifti" and f.origin.type == "job":
                        # checking origin file in case there are multiple nifti
                        job = self.client.get_job(f.origin.id)
                        if job.inputs["dcm2niix_input"].get("name") == in_file_ref.get(
                            "name"
                        ):
                            source_nifti = f
                            break
                if not source_nifti:
                    log.error(
                        f"No NIfTI source file found for analysis {ana.label} {ana.ref()} - SKIPPING"
                    )
                    continue
                else:
                    ana.update_info({"source_nifti": source_nifti.ref()})
                    ana.update_info({"source_acq": {"label": acq.label, **acq.ref()}})
                    log.info(f"...DONE")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Add NIfTI source reference to roi2nix analysis container custom "
                    "information"
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
