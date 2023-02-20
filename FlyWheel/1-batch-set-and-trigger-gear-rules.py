#!/usr/bin/env python
"""
A hierarchy-curator that set gear rule on project and fiddle file type for DICOM files
to re trigger gear rules.
"""
import argparse
import logging

import flywheel
from flywheel_gear_toolkit.utils.curator import HierarchyCurator
from flywheel_gear_toolkit.utils.walker import Walker


def add_file_metadata_importer(client: flywheel.Client, project: flywheel.Project):
    gear_name = "file-metadata-importer"
    log.info(f"Adding gear rule for {gear_name}...")
    gear = client.lookup(f"gears/{gear_name}")
    rule = flywheel.models.rule.Rule(
        project_id=project.id,
        **{
            "all": [{"regex": False, "type": "file.type", "value": "dicom"}],
            "_not": [],
            "any": [],
        },
        gear_id=gear.id,
        name=gear.gear.name,
        auto_update=False,
        disabled=False,
    )
    client.add_project_rule(project.id, rule)


def add_file_classifier(client: flywheel.Client, project: flywheel.Project):
    gear_name = "file-classifier"
    log.info(f"Adding gear rule for {gear_name}...")
    gear = client.lookup(f"gears/{gear_name}")
    rule = flywheel.models.rule.Rule(
        project_id=project.id,
        **{
            "all": [
                {"regex": False, "type": "file.type", "value": "dicom"},
                {
                    "regex": False,
                    "type": "file.tags",
                    "value": "file-metadata-importer",
                },
            ],
            "_not": [],
            "any": [],
        },
        gear_id=gear.id,
        name=gear.gear.name,
        auto_update=False,
        disabled=False,
    )
    client.add_project_rule(project.id, rule)


def add_dcm2niix(client: flywheel.Client, project: flywheel.Project):
    gear_name = "dcm2niix"
    log.info(f"Adding gear rule for {gear_name}...")
    gear = client.lookup(f"gears/{gear_name}")
    rule = flywheel.models.rule.Rule(
        project_id=project.id,
        **{
            "all": [
                {"regex": False, "type": "file.type", "value": "dicom"},
            ],
            "_not": [
                {"regex": False, "type": "file.classification", "value": "Non-Image"}
            ],
            "any": [],
        },
        gear_id=gear.id,
        name=gear.gear.name,
        auto_update=False,
        disabled=False,
    )
    client.add_project_rule(project.id, rule)


class Curator(HierarchyCurator):
    def curate_project(self, project):
        gear_rules = self.client.get_project_rules(project.id)
        gear_rules_l = [
            self.client.get_gear(g.gear_id)["gear"]["name"] for g in gear_rules
        ]
        # we are assuming simple logic for checking gear existence based only
        # on presence of a gear rule based on that a certain gear name
        if "file-metadata-importer" not in gear_rules_l:
            add_file_metadata_importer(self.client, project)
        if "file-classifier" not in gear_rules_l:
            add_file_classifier(self.client, project)
        if "dcm2niix" not in gear_rules_l:
            add_dcm2niix(self.client, project)

    def curate_file(self, file):
        # fiddle file.type for DICOM files to trigger gear rules
        #print(f'\n\n file.name = {file.name}, \n\t file.classification = {file.classification} \n\t file.type= {file.type} \n\n\n')
        #print(file.__dict__)
        #print(f'file.modality={file.modality}')
        #print(f'file.type={file.type}\n\n\n')
        
        
        file.update(modality='CT')
            
        if file.type == "dicom":
            log.info(f"Fiddling file.type for {file.ref()}")
            file.update(type="document")
            file.update(type="dicom")
            
        else:
            print(f'Error not the right claissfier: {file.type}')
            print(f'\n\n file.name = {file.name}, \n\t file.classification = {file.classification} \n\t file.type= {file.type} \n\n\n')
            file.update(type="dicom")
            
            
            
            
            
            file.add_tag("Triggered_dicom")
            
            
            #file.update(classification=None)
            #file.update(classification={'Intent': ['Localizer'], 'Measurement': ['T2']})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Set gear rule on project and fiddle file type for DICOM files to "
                    "re trigger gear rules."
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
